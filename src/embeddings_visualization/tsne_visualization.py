import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from down_tasks.forecast_chronos import (
    build_dataset_group as build_chronos_dataset_group,
    _parse_dataset_names as parse_chronos_dataset_names,
)
from down_tasks.forecast_shared import apply_model_overrides, parse_horizon_values
from time_series_loader import TimeSeriesDataModule
from util import (
    default_device,
    extract_sequence,
    load_encoder_checkpoint,
    prepare_sequence,
)


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def slugify_label(value: str) -> str:
    slug = _SLUG_PATTERN.sub("_", value).strip("_").lower()
    return slug or "dataset"


def resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    if candidate is None:
        return None
    candidate_path = Path(str(candidate)).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = (base / candidate_path).resolve()
    else:
        candidate_path = candidate_path.resolve()
    return candidate_path


def load_yaml_mapping(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def discover_directory_names(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted(
        child.name for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")
    )


def build_icml_dataset_loaders(
    data_dir: Path,
    *,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    normalize: bool,
    selected: Optional[Sequence[str]] = None,
) -> List[Tuple[str, DataLoader]]:
    module = TimeSeriesDataModule(
        dataset_name="",
        data_dir=str(data_dir),
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize,
        train=True,
        val=False,
        test=False,
        grouped=True,
    )
    groups = module.get_dataloaders()
    allowed = {entry.lower() for entry in selected} if selected else None
    loaders: List[Tuple[str, DataLoader]] = []
    for group in groups:
        loader = group.train
        if loader is None:
            continue
        normalized_name = group.name.replace("\\", "/")
        group_path = Path(normalized_name)
        folder = group_path.parts[0].lower() if group_path.parts else group_path.stem.lower()
        stem = group_path.stem.lower()
        if allowed is not None and not (
            normalized_name.lower() in allowed or folder in allowed or stem in allowed
        ):
            continue
        loaders.append((normalized_name, loader))
    return loaders


def build_chronos_dataset_loaders(
    dataset_names: Sequence[str],
    *,
    repo_id: str,
    split: str,
    target_dtype: Optional[str],
    normalize_per_series: bool,
    load_kwargs: Dict[str, object],
    context_length: int,
    horizon: int,
    stride: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    torch_dtype: torch.dtype,
    max_windows_per_series: Optional[int],
    max_series: Optional[int],
    val_ratio: float,
    seed: int,
) -> List[Tuple[str, DataLoader]]:
    loaders: List[Tuple[str, DataLoader]] = []
    for dataset_name in dataset_names:
        try:
            group = build_chronos_dataset_group(
                dataset_name,
                repo_id=repo_id,
                split=split,
                target_dtype=target_dtype,
                normalize_per_series=normalize_per_series,
                load_kwargs=dict(load_kwargs),
                context_length=context_length,
                horizon=horizon,
                stride=stride,
                batch_size=batch_size,
                val_batch_size=val_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                torch_dtype=torch_dtype,
                max_windows_per_series=max_windows_per_series,
                max_series=max_series,
                val_ratio=val_ratio,
                seed=seed,
            )
        except Exception as exc:
            print(f"Warning: failed to build Chronos loader for {dataset_name}: {exc}")
            continue
        if group is None or group.train is None:
            print(f"Warning: Chronos dataset {dataset_name} produced no train loader; skipping.")
            continue
        loaders.append((dataset_name, group.train))
    return loaders


def extract_embeddings_from_loader(
    loader: DataLoader,
    *,
    encoder: torch.nn.Module,
    device: torch.device,
    max_samples: Optional[int],
) -> Tuple[Optional[np.ndarray], torch.device]:
    limit = max_samples if max_samples is not None and max_samples > 0 else None
    collected = 0
    chunks: List[np.ndarray] = []
    active_device = device
    oom_reported = False

    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            seq = prepare_sequence(extract_sequence(batch)).to(device=active_device, dtype=torch.float32)
            seq = seq.transpose(1, 2)
            try:
                outputs = encoder(seq)
            except RuntimeError as exc:
                message = str(exc).lower()
                if active_device.type == "cuda" and "out of memory" in message:
                    if not oom_reported:
                        print("CUDA OOM encountered during embedding extraction; switching encoder to CPU.")
                        oom_reported = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    encoder.to("cpu")
                    active_device = torch.device("cpu")
                    seq = seq.to(active_device)
                    outputs = encoder(seq)
                else:
                    raise

            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            embeddings_cpu = outputs.detach().to("cpu")
            chunk = embeddings_cpu.numpy()
            if limit is not None and collected + chunk.shape[0] > limit:
                chunk = chunk[: limit - collected]
            chunks.append(chunk)
            collected += chunk.shape[0]

            if active_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            del seq, outputs

            if limit is not None and collected >= limit:
                break

    if not chunks:
        return None, active_device
    return np.concatenate(chunks, axis=0), active_device


def discover_available_datasets(
    chronos_data_dir: Path,
    icml_data_dir: Path,
) -> Tuple[List[str], List[str]]:
    chronos = discover_directory_names(chronos_data_dir)
    icml = discover_directory_names(icml_data_dir)
    print(
        f"Found {len(chronos)} Chronos datasets: {chronos[:5]}{'...' if len(chronos) > 5 else ''}"
    )
    print(f"Found {len(icml)} ICML datasets: {icml}")
    return chronos, icml


def get_encoder_path(config_path: Path) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    paths_section = cfg.get("paths") or {}
    encoder_path = paths_section.get("encoder_checkpoint")
    if encoder_path is None:
        raise KeyError("Configuration missing 'paths.encoder_checkpoint'.")
    candidate = Path(str(encoder_path))
    resolved = candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()
    return str(resolved)


def tsne_visualization(
    config_path: str,
    chronos_datasets: Optional[Sequence[str]] = None,
    icml_datasets: Optional[Sequence[str]] = None,
    repo_id: str = "autogluon/chronos_datasets",
    output_dir: str = "results/tsne",
    max_samples_per_dataset: Optional[int] = 500,
    batch_size: int = 128,
    chronos_data_dir: str = "../../data",
    icml_data_dir: str = "../../ICML_datasets",
) -> None:
    print("Starting t-SNE visualization")
    cfg_path = Path(config_path).expanduser().resolve()
    encoder_path = Path(get_encoder_path(cfg_path))
    print(f"Encoder path: {encoder_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        forecast_cfg = yaml.safe_load(handle) or {}

    model_section = dict(forecast_cfg.get("model") or {})
    model_config_candidate = model_section.get("config")
    if model_config_candidate is None:
        raise ValueError("Configuration missing required key 'model.config'.")
    model_config_path = Path(str(model_config_candidate))
    if not model_config_path.is_absolute():
        model_config_path = (cfg_path.parent / model_config_path).resolve()

    base_config = tu.load_config(model_config_path)
    overrides = dict(model_section.get("overrides") or {})
    model_cfg = apply_model_overrides(
        base_config.model,
        token_size=overrides.get("token_size"),
        model_dim=overrides.get("model_dim"),
        embedding_dim=overrides.get("embedding_dim"),
        depth=overrides.get("depth"),
    )

    training_section = dict(forecast_cfg.get("training") or {})
    horizons_raw = training_section.get("horizons", [])
    if isinstance(horizons_raw, str):
        horizons = parse_horizon_values(horizons_raw)
    else:
        horizons = [int(h) for h in horizons_raw]
    if not horizons:
        raise ValueError("No forecast horizons configured.")
    max_horizon = max(horizons)

    data_cfg = dict(base_config.data or {})
    seed = int(training_section.get("seed", base_config.seed))
    icml_batch_size = int(training_section.get("batch_size", data_cfg.get("batch_size", batch_size)))
    icml_val_batch_size = int(
        training_section.get("val_batch_size", data_cfg.get("val_batch_size", icml_batch_size))
    )
    num_workers = int(training_section.get("num_workers", data_cfg.get("num_workers", 0)))
    pin_memory = bool(training_section.get("pin_memory", data_cfg.get("pin_memory", True)))
    normalize_icml = bool(data_cfg.get("normalize", True))

    chronos_base_dir = Path(chronos_data_dir).expanduser()
    if not chronos_base_dir.is_absolute():
        chronos_base_dir = (cfg_path.parent / chronos_base_dir).resolve()
    icml_base_dir = Path(icml_data_dir).expanduser()
    if not icml_base_dir.is_absolute():
        icml_base_dir = (cfg_path.parent / icml_base_dir).resolve()

    chronos_available, _icml_available = discover_available_datasets(chronos_base_dir, icml_base_dir)

    chronos_section = dict(forecast_cfg.get("chronos") or {})
    chronos_cfg: Dict[str, object] = {}
    loader_config_candidate = chronos_section.get("config")
    if loader_config_candidate is not None:
        loader_config_path = resolve_optional_path(cfg_path.parent, loader_config_candidate)
        if loader_config_path is None or not loader_config_path.exists():
            raise FileNotFoundError(f"Chronos loader config not found: {loader_config_candidate}")
        chronos_cfg.update(load_yaml_mapping(loader_config_path))

    resolved_chronos = list(chronos_datasets) if chronos_datasets is not None else []
    if not resolved_chronos:
        config_list = parse_chronos_dataset_names(
            chronos_section.get("datasets") or chronos_cfg.get("datasets") or chronos_cfg.get("datasets_to_load")
        )
        if config_list:
            resolved_chronos = list(config_list)
        else:
            resolved_chronos = chronos_available

    icml_selected = list(icml_datasets) if icml_datasets is not None else None

    chronos_split = str(chronos_section.get("split", chronos_cfg.get("split", "train")))
    target_dtype = chronos_section.get("target_dtype", chronos_cfg.get("target_dtype"))
    normalize_per_series = bool(chronos_section.get("normalize", chronos_cfg.get("normalize", True)))
    val_ratio = float(chronos_section.get("val_split", chronos_cfg.get("val_split", 0.2)))

    context_length = chronos_section.get("context_length")
    if context_length is None:
        context_length = chronos_cfg.get("context_length", chronos_cfg.get("patch_length", max_horizon))
    context_length = int(context_length)

    stride_value = chronos_section.get("window_stride", chronos_cfg.get("window_stride"))
    stride = int(stride_value) if stride_value is not None else max_horizon
    stride = max(1, stride)

    max_windows_per_series = chronos_section.get(
        "max_windows_per_series", chronos_cfg.get("max_windows_per_series")
    )
    if max_windows_per_series is not None:
        max_windows_per_series = int(max_windows_per_series)

    max_series = chronos_section.get("max_series", chronos_cfg.get("max_series"))
    if max_series is not None:
        max_series = int(max_series)

    load_kwargs: Dict[str, object] = dict(chronos_cfg.get("load_kwargs", {}) or {})
    section_load_kwargs = chronos_section.get("load_kwargs")
    if isinstance(section_load_kwargs, dict):
        load_kwargs.update(section_load_kwargs)

    offline_cache_dir = resolve_optional_path(cfg_path.parent, load_kwargs.get("offline_cache_dir"))
    if offline_cache_dir is None:
        offline_cache_dir = chronos_base_dir
    load_kwargs["offline_cache_dir"] = str(offline_cache_dir)
    if "force_offline" not in load_kwargs:
        load_kwargs["force_offline"] = True

    device = default_device()
    print(f"Using device: {device}")
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    print(f"Loading encoder checkpoint: {encoder_path}")
    load_encoder_checkpoint(encoder, encoder_path, device)
    encoder.eval()

    output_dir_path = Path(output_dir).expanduser()
    if not output_dir_path.is_absolute():
        output_dir_path = (ROOT_DIR / output_dir_path).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir_path / f"embeddings_{encoder_path.stem}"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    for old_file in embeddings_dir.glob("*.npy"):
        try:
            old_file.unlink()
        except OSError as exc:
            print(f"Warning: failed to remove stale embedding file {old_file}: {exc}")

    chronos_loaders = build_chronos_dataset_loaders(
        resolved_chronos,
        repo_id=repo_id,
        split=chronos_split,
        target_dtype=target_dtype,
        normalize_per_series=normalize_per_series,
        load_kwargs=load_kwargs,
        context_length=context_length,
        horizon=max_horizon,
        stride=stride,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        torch_dtype=torch.float32,
        max_windows_per_series=max_windows_per_series,
        max_series=max_series,
        val_ratio=val_ratio,
        seed=seed,
    )

    icml_loaders = build_icml_dataset_loaders(
        icml_base_dir,
        batch_size=icml_batch_size,
        val_batch_size=icml_val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize_icml,
        selected=icml_selected,
    )

    dataset_entries: List[Tuple[str, str, DataLoader]] = []
    dataset_entries.extend(("Chronos", name, loader) for name, loader in chronos_loaders)
    dataset_entries.extend(("ICML", name, loader) for name, loader in icml_loaders)

    if not dataset_entries:
        raise RuntimeError("No datasets available for embedding extraction.")

    embedding_records: List[Tuple[str, Path, int]] = []
    dataset_labels: List[str] = []

    for source, name, loader in dataset_entries:
        label = f"{source}: {name}"
        print(f"Extracting embeddings for {label}")
        embeddings, device = extract_embeddings_from_loader(
            loader,
            encoder=encoder,
            device=device,
            max_samples=max_samples_per_dataset,
        )
        if embeddings is None or embeddings.size == 0:
            print(f"Warning: no embeddings extracted for {label}")
            continue
        sample_count = int(embeddings.shape[0])
        slug = slugify_label(label)
        file_path = embeddings_dir / f"{len(embedding_records):03d}_{slug}.npy"
        np.save(file_path, embeddings)
        embedding_records.append((label, file_path, sample_count))
        print(f"Collected {sample_count} embeddings for {label} -> {file_path}")

    if not embedding_records:
        raise RuntimeError("No embeddings produced for any dataset.")

    print(f"Embedding files written to {embeddings_dir}")

    dataset_labels = [record[0] for record in embedding_records]

    loaded_embeddings: List[np.ndarray] = []
    combined_labels: List[int] = []
    for label_index, (dataset_label, file_path, expected_count) in enumerate(embedding_records):
        data = np.load(file_path, allow_pickle=False)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        actual_count = int(data.shape[0])
        if expected_count != actual_count:
            print(
                f"Warning: embedding count mismatch for {dataset_label}; expected {expected_count}, reloaded {actual_count}."
            )
        loaded_embeddings.append(data)
        combined_labels.extend([label_index] * actual_count)

    combined_embeddings = np.concatenate(loaded_embeddings, axis=0)
    combined_labels = np.array(combined_labels, dtype=np.int64)
    total_samples = combined_embeddings.shape[0]
    print(f"Total embeddings: {total_samples}")

    perplexity = max(2, min(30, total_samples // 3))
    if perplexity >= total_samples:
        perplexity = max(2, total_samples - 1)
    print(f"Running t-SNE with perplexity={perplexity}")

    tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=perplexity, max_iter=1000)
    tsne_results = tsne.fit_transform(combined_embeddings)
    print("t-SNE completed")

    plt.figure(figsize=(15, 12))
    cmap = plt.get_cmap("tab20", max(1, len(dataset_labels)))

    for label_index, dataset_label in enumerate(dataset_labels):
        indices = np.where(combined_labels == label_index)[0]
        if indices.size == 0:
            continue
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            label=f"{dataset_label} (n={indices.size})",
            alpha=0.7,
            s=20,
            color=cmap(label_index),
        )

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.title("t-SNE Visualization of Time Series Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()

    encoder_name = encoder_path.stem
    plot_path = output_dir_path / f"tsne_{encoder_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"t-SNE plot saved to {plot_path}")


if __name__ == "__main__":
    default_config = ROOT_DIR / "src" / "configs" / "chronos_forecast.yaml"
    try:
        tsne_visualization(
            config_path=str(default_config),
            chronos_datasets=None,
            icml_datasets=None,
            repo_id="autogluon/chronos_datasets",
            output_dir="results/tsne",
            max_samples_per_dataset=200,
            batch_size=64,
            chronos_data_dir="../../data",
            icml_data_dir="../../ICML_datasets",
        )
    except Exception as exc:
        print(f"Error during t-SNE visualization: {exc}")
        import traceback

        traceback.print_exc()
