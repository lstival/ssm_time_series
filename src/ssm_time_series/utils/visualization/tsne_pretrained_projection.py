"""Project Chronos and TimesFM embeddings via t-SNE."""

from __future__ import annotations

# Removed legacy sys.path hack

from ssm_time_series import training as tu
from embeddings_visualization.projection_utils import (
    build_chronos_dataset_groups,
    build_output_dir,
    determine_config_path,
    fit_tsne,
    infer_dataset_type,
    load_projection_config,
)
from ssm_time_series.utils.nn import default_device

TSNE_HF_CONFIG_ENV_VAR = "TSNE_HF_CONFIG"
DEFAULT_TSNE_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection.yaml"
DEFAULT_MODEL_CONFIG_PATH = SRC_DIR / "configs" / "mamba_encoder.yaml"
DEFAULT_OUTPUT_PREFIX = "tsne_hf_projection"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "tsne_hf"
DEFAULT_DATA_DIR = ROOT_DIR / "chronos"
DEFAULT_SPLIT = "all"

DEFAULT_VARIANTS: List[Dict[str, str]] = [
    {
        "label": "Chronos-T5-Small",
        "kind": "chronos",
        "model_id": "amazon/chronos-t5-small",
    },
    {
        "label": "TimesFM-2.0-500M",
        "kind": "timesfm",
        "model_id": "google/timesfm-2.0-500m-pytorch",
    },
]



@dataclass
class HFVariantSpec:
    label: str
    kind: str
    model_id: str


class ChronosEncoder:
    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=self.device,
            dtype=self.dtype,
        )

    def encode(self, series: torch.Tensor) -> torch.Tensor:
        x_cpu = series.detach().cpu().float()
        result = self.pipeline.embed(x_cpu)
        if isinstance(result, tuple):
            embeddings = result[0]
        else:
            embeddings = result
        embeddings = embeddings.to(device=self.device, dtype=self.dtype)
        if embeddings.dim() > 2:
            embeddings = embeddings.mean(dim=1)
        return embeddings


class TimesFmEncoder:
    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        self.model = TimesFmModelForPrediction.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

    def encode(self, series: torch.Tensor) -> torch.Tensor:
        inputs = series.to(self.device, dtype=torch.float32)
        outputs = self.model(past_values=inputs, return_dict=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
        if hidden is None:
            raise RuntimeError("TimesFM model did not return last_hidden_state")
        embeddings = hidden
        if embeddings.dim() == 3:
            embeddings = embeddings[:, -1, :]
        elif embeddings.dim() > 3:
            embeddings = embeddings.flatten(start_dim=1)
        return embeddings


def slugify_label(label: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in label]
    slug = "".join(cleaned).strip("_")
    return slug or "model"


def load_variant_specs(config_path: Path) -> List[HFVariantSpec]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    hf_section = payload.get("huggingface") or payload.get("hf") or {}
    raw_variants = hf_section.get("variants")

    specs: List[HFVariantSpec] = []
    if isinstance(raw_variants, list):
        for entry in raw_variants:
            if isinstance(entry, dict):
                model_id = entry.get("model_id")
                if not model_id:
                    print("Skipping Hugging Face variant without model_id.")
                    continue
                label = entry.get("label") or entry.get("model_id") or "model"
                kind = str(entry.get("kind", "")).lower()
                specs.append(HFVariantSpec(label=str(label), kind=kind, model_id=str(model_id)))
            elif isinstance(entry, str):
                specs.append(HFVariantSpec(label=entry, kind="chronos", model_id=entry))

    if not specs:
        specs = [
            HFVariantSpec(label=item["label"], kind=item["kind"], model_id=item["model_id"])
            for item in DEFAULT_VARIANTS
        ]

    return specs


def extract_primary_channel(
    tensor: torch.Tensor,
    *,
    dataset_label: str,
    sequence_first_input: bool,
    warned_datasets: Set[str],
) -> torch.Tensor:
    if tensor.dim() == 3:
        if sequence_first_input:
            feature_dim = tensor.size(-1)
            if feature_dim > 1 and dataset_label not in warned_datasets:
                print(
                    f"  Warning: dataset '{dataset_label}' has {feature_dim} features; using the first feature only."
                )
                warned_datasets.add(dataset_label)
            series = tensor[..., 0]
        else:
            feature_dim = tensor.size(1)
            if feature_dim > 1 and dataset_label not in warned_datasets:
                print(
                    f"  Warning: dataset '{dataset_label}' has {feature_dim} features; using the first feature only."
                )
                warned_datasets.add(dataset_label)
            series = tensor[:, 0, :]
    elif tensor.dim() == 2:
        series = tensor
    else:
        raise ValueError("Unsupported tensor shape for time series input")
    return series.contiguous()


def collect_variant_embeddings(
    *,
    encoder,
    loader,
    dataset_label: str,
    samples_per_dataset: int,
    sequence_first_input: bool,
    warned_datasets: Set[str],
) -> Tuple[np.ndarray, List[int]]:
    collected: List[torch.Tensor] = []
    sample_indices: List[int] = []
    local_index = 0

    with torch.no_grad():
        for batch in loader:
            series = extract_primary_channel(
                batch[0].float(),
                dataset_label=dataset_label,
                sequence_first_input=sequence_first_input,
                warned_datasets=warned_datasets,
            )
            embeddings = encoder.encode(series)
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            embeddings = embeddings.detach().to(device="cpu", dtype=torch.float32)
            if embeddings.dim() > 2:
                embeddings = embeddings.flatten(start_dim=1)
            collected.append(embeddings)

            batch_size = embeddings.size(0)
            sample_indices.extend(range(local_index, local_index + batch_size))
            local_index += batch_size
            if local_index >= samples_per_dataset:
                break

    if not collected:
        return np.empty((0, 0), dtype=np.float32), []

    stacked = torch.cat(collected, dim=0)
    if stacked.size(0) > samples_per_dataset:
        stacked = stacked[:samples_per_dataset]
        sample_indices = sample_indices[:samples_per_dataset]

    return stacked.numpy(), sample_indices


def instantiate_encoder(spec: HFVariantSpec, device: torch.device):
    if spec.kind == "chronos":
        return ChronosEncoder(spec.model_id, device)
    if spec.kind in {"timesfm", "times_fm"}:
        return TimesFmEncoder(spec.model_id, device)
    raise ValueError(f"Unknown Hugging Face variant kind: {spec.kind}")


def run_variant_projection(
    *,
    spec: HFVariantSpec,
    dataset_groups,
    device: torch.device,
    proj_cfg,
    results_dir: Path,
    seed: Optional[int],
    warned_datasets: Set[str],
    projection_config_path: Path,
) -> None:
    print(f"\n=== Running projection for {spec.label} ({spec.kind}) ===")
    encoder = instantiate_encoder(spec, device)

    dataset_entries: List[Dict[str, object]] = []
    for group in dataset_groups:
        loader = group.loader
        dataset_label = group.name
        dataset_type = infer_dataset_type(dataset_label)
        split_used = group.split

        try:
            embeddings, sample_indices = collect_variant_embeddings(
                encoder=encoder,
                loader=loader,
                dataset_label=dataset_label,
                samples_per_dataset=proj_cfg.samples_per_dataset,
                sequence_first_input=proj_cfg.sequence_first_input,
                warned_datasets=warned_datasets,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"  Skipping dataset '{dataset_label}' due to runtime error: {exc}")
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            continue

        sample_total = embeddings.shape[0]
        if sample_total == 0:
            print(f"  Dataset '{dataset_label}' produced no samples; skipping.")
            continue

        dataset_entries.append(
            {
                "name": dataset_label,
                "type": dataset_type,
                "split": split_used,
                "embeddings": embeddings.astype(np.float32, copy=False),
                "indices": sample_indices,
            }
        )
        print(f"  Collected {sample_total} embeddings for '{dataset_label}' ({dataset_type}).")

    if not dataset_entries:
        print("No embeddings collected; skipping t-SNE for this variant.")
        return

    embedding_blocks = [entry["embeddings"] for entry in dataset_entries]
    embedding_matrix = np.concatenate(embedding_blocks, axis=0)

    coords = fit_tsne(
        embedding_matrix,
        perplexity=proj_cfg.perplexity,
        learning_rate=proj_cfg.learning_rate,
        n_iter=proj_cfg.n_iter,
        seed=seed,
    )

    records: List[Dict[str, object]] = []
    cursor = 0
    for entry in dataset_entries:
        embeddings = entry["embeddings"]
        count = embeddings.shape[0]
        indices = entry["indices"]
        for idx in range(count):
            records.append(
                {
                    "model": spec.label,
                    "model_kind": spec.kind,
                    "dataset": entry["name"],
                    "dataset_type": entry["type"],
                    "split": entry["split"],
                    "sample_in_dataset": indices[idx] if idx < len(indices) else idx,
                    "global_index": cursor + idx,
                }
            )
        cursor += count

    coords_df = pd.DataFrame.from_records(records)
    coords_df["tsne_x"] = coords[:, 0]
    coords_df["tsne_y"] = coords[:, 1]

    slug = slugify_label(spec.label)
    model_prefix = f"{proj_cfg.output_prefix}_{slug}"
    output_dir = build_output_dir(results_dir, model_prefix, None)
    csv_path = output_dir / "tsne_coordinates.csv"
    plot_path = output_dir / "tsne_plot.png"

    coords_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_types = sorted(coords_df["dataset_type"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_types))))

    for color, dataset_type in zip(colors, unique_types):
        mask = coords_df["dataset_type"] == dataset_type
        ax.scatter(
            coords_df.loc[mask, "tsne_x"],
            coords_df.loc[mask, "tsne_y"],
            label=dataset_type,
            s=16,
            alpha=0.75,
            color=color,
        )

    ax.set_title(f"{spec.label} Embeddings (t-SNE)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)

    if proj_cfg.show:
        plt.show()
    plt.close(fig)

    print("\nSaved t-SNE artifacts:")
    print(f"  Coordinates CSV: {csv_path}")
    print(f"  Plot:            {plot_path}")
    print("\nModel references:")
    print(f"  TSNE config:        {projection_config_path}")
    print(f"  Hugging Face model: {spec.model_id}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    projection_config_path = determine_config_path(
        env_var=TSNE_HF_CONFIG_ENV_VAR,
        default_path=DEFAULT_TSNE_CONFIG_PATH,
    )
    proj_cfg = load_projection_config(
        projection_config_path,
        default_model_config_path=DEFAULT_MODEL_CONFIG_PATH,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        default_output_prefix=DEFAULT_OUTPUT_PREFIX,
        default_data_dir=DEFAULT_DATA_DIR,
        default_split=DEFAULT_SPLIT,
    )

    base_config = tu.load_config(proj_cfg.model_config_path)

    seed = proj_cfg.seed if proj_cfg.seed is not None else getattr(base_config, "seed", None)
    if seed is not None:
        tu.set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    device_spec = proj_cfg.device.strip().lower()
    if device_spec == "auto":
        device = default_device()
    else:
        device = torch.device(proj_cfg.device)
    print(f"Using device: {device}")

    results_dir = proj_cfg.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = load_variant_specs(projection_config_path)
    if not variants:
        raise RuntimeError("No Hugging Face variants configured for projection")

    dataset_groups = build_chronos_dataset_groups(proj_cfg=proj_cfg)
    if not dataset_groups:
        raise RuntimeError("No Chronos datasets available for visualization")

    warned_datasets: Set[str] = set()
    for spec in variants:
        run_variant_projection(
            spec=spec,
            dataset_groups=dataset_groups,
            device=device,
            proj_cfg=proj_cfg,
            results_dir=results_dir,
            seed=seed,
            warned_datasets=warned_datasets,
            projection_config_path=projection_config_path,
        )


if __name__ == "__main__":
    main()
