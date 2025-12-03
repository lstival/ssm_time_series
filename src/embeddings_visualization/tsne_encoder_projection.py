"""Project ICML encoder embeddings via t-SNE with dataset-type coloring."""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from evaluation_down_tasks.zeroshot_utils import (
    extract_checkpoint_timestamp,
    select_loader,
)
from time_series_loader import TimeSeriesDataModule
from util import build_projection_head, default_device, load_encoder_checkpoint

TSNE_CONFIG_ENV_VAR = "TSNE_ENCODER_CONFIG"
DEFAULT_TSNE_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection.yaml"
DEFAULT_MODEL_CONFIG_PATH = SRC_DIR / "configs" / "mamba_encoder.yaml"
DEFAULT_OUTPUT_PREFIX = "tsne_encoder_projection"
DEFAULT_SPLIT = "all"


@dataclass
class ProjectionConfig:
    config_path: Path
    model_config_path: Path
    encoder_checkpoint: Path
    projection_checkpoint: Optional[Path]
    output_dir: Path
    output_prefix: str
    samples_per_dataset: int
    perplexity: float
    learning_rate: float
    n_iter: int
    seed: Optional[int]
    device: str
    show: bool
    data_dir: Path
    dataset_names: Optional[List[str]]
    dataset_name: str
    filename: Optional[Path]
    batch_size: int
    val_batch_size: int
    num_workers: int
    split: str
    sequence_first_input: bool


def _determine_projection_config_path() -> Path:
    env_value = os.environ.get(TSNE_CONFIG_ENV_VAR)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return candidate
    default_path = DEFAULT_TSNE_CONFIG_PATH.expanduser().resolve()
    if default_path.exists():
        return default_path
    raise FileNotFoundError(
        "Projection config not found. Provide TSNE config via TSNE_ENCODER_CONFIG env variable or create configs/tsne_encoder_projection.yaml"
    )


def _resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    if candidate is None:
        return None
    path_candidate = Path(str(candidate)).expanduser()
    if not path_candidate.is_absolute():
        path_candidate = (base / path_candidate).resolve()
    else:
        path_candidate = path_candidate.resolve()
    return path_candidate


def _resolve_required_path(base: Path, candidate: Optional[object], *, description: str) -> Path:
    path = _resolve_optional_path(base, candidate)
    if path is None:
        raise ValueError(f"Missing required path for {description}")
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _load_projection_config(config_path: Path) -> ProjectionConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    base_dir = config_path.parent
    paths_section = dict(payload.get("paths") or {})
    tsne_section = dict(payload.get("tsne") or {})
    data_section = dict(payload.get("data") or {})

    model_config_path = _resolve_optional_path(base_dir, paths_section.get("model_config"))
    if model_config_path is None:
        model_config_path = DEFAULT_MODEL_CONFIG_PATH
    model_config_path = model_config_path.expanduser().resolve()
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    encoder_checkpoint = _resolve_required_path(
        base_dir,
        paths_section.get("encoder_checkpoint"),
        description="encoder checkpoint",
    )
    projection_checkpoint = _resolve_optional_path(base_dir, paths_section.get("projection_checkpoint"))
    output_dir = _resolve_optional_path(base_dir, paths_section.get("output_dir"))
    if output_dir is None:
        output_dir = (ROOT_DIR / "results" / "tsne_encoder").resolve()

    data_dir_candidate = data_section.get("data_dir")
    if data_dir_candidate is None:
        data_dir_candidate = ROOT_DIR / "ICML_datasets"
    data_dir = _resolve_required_path(base_dir, data_dir_candidate, description="data directory")

    dataset_names_value = data_section.get("dataset_names")
    if isinstance(dataset_names_value, list):
        dataset_names = [str(name) for name in dataset_names_value]
    elif isinstance(dataset_names_value, str) and dataset_names_value.strip():
        dataset_names = [dataset_names_value.strip()]
    else:
        dataset_names = None

    dataset_name = str(data_section.get("dataset_name") or "").strip()
    filename = _resolve_optional_path(base_dir, data_section.get("filename"))

    batch_size = int(data_section.get("batch_size", 512))
    val_batch_size = int(data_section.get("val_batch_size", batch_size))
    num_workers = int(data_section.get("num_workers", 4))
    split = str(data_section.get("split", DEFAULT_SPLIT)).lower()

    output_prefix = str(tsne_section.get("output_prefix", DEFAULT_OUTPUT_PREFIX))
    sequence_first = bool(tsne_section.get("sequence_first_input", False))

    seed_value = tsne_section.get("seed")
    seed_int = int(seed_value) if seed_value is not None else None

    return ProjectionConfig(
        config_path=config_path,
        model_config_path=model_config_path,
        encoder_checkpoint=encoder_checkpoint,
        projection_checkpoint=projection_checkpoint,
        output_dir=output_dir,
        output_prefix=output_prefix,
        samples_per_dataset=int(tsne_section.get("samples_per_dataset", 1000)),
        perplexity=float(tsne_section.get("perplexity", 35.0)),
        learning_rate=float(tsne_section.get("learning_rate", 200.0)),
        n_iter=int(tsne_section.get("n_iter", 1000)),
        seed=seed_int,
        device=str(tsne_section.get("device", "auto")),
        show=bool(tsne_section.get("show", False)),
        data_dir=data_dir,
        dataset_names=dataset_names,
        dataset_name=dataset_name,
        filename=filename,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        split=split,
        sequence_first_input=sequence_first,
    )


def _load_projection_head(
    *,
    encoder: torch.nn.Module,
    projection_checkpoint_path: Optional[Path],
    device: torch.device,
) -> Optional[torch.nn.Module]:
    if projection_checkpoint_path is None:
        print("Warning: projection checkpoint not provided; using encoder embeddings directly.")
        return None
    projection_ckpt_path = projection_checkpoint_path.expanduser().resolve()
    if not projection_ckpt_path.exists():
        print(
            f"Warning: projection head checkpoint not found at {projection_ckpt_path}; proceeding with raw encoder embeddings."
        )
        return None

    projection_head = build_projection_head(encoder).to(device)
    projection_head.eval()

    checkpoint = torch.load(projection_ckpt_path, map_location=device)
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model_state"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
    if not state_dict:
        raise ValueError(f"Projection checkpoint at {projection_ckpt_path} does not contain weights")

    missing, unexpected = projection_head.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing projection head weights: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected projection head weights: {sorted(unexpected)}")

    return projection_head


def _infer_dataset_type(name: str) -> str:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized:
        return "dataset"
    parts = normalized.split("/")
    base = parts[0] if len(parts) > 1 else normalized
    base_path = Path(base)
    stem = base_path.stem if base_path.suffix else base_path.name
    return stem or "dataset"


def _collect_embeddings_for_dataset(
    *,
    encoder: torch.nn.Module,
    projection_head: Optional[torch.nn.Module],
    loader,
    device: torch.device,
    samples_per_dataset: int,
    sequence_first_input: bool,
) -> Tuple[np.ndarray, List[int]]:
    if loader is None:
        return np.empty((0, 0), dtype=np.float32), []

    collected: List[torch.Tensor] = []
    sample_indices: List[int] = []
    encoder.eval()
    local_index = 0

    with torch.no_grad():
        for batch in loader:
            seq_x = batch[0].float().to(device)
            if not sequence_first_input:
                seq_x = seq_x.transpose(1, 2)

            embeddings = encoder(seq_x)
            if projection_head is not None:
                embeddings = projection_head(embeddings)
            if embeddings.dim() > 2:
                embeddings = embeddings.flatten(start_dim=1)
            embeddings = embeddings.detach().cpu()
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


def _instantiate_tsne(
    *,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    seed: Optional[int],
):
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": learning_rate,
        "init": "pca",
        "random_state": seed,
    }
    try:
        params = inspect.signature(TSNE.__init__).parameters
    except (ValueError, TypeError):
        params = {}
    if "n_iter" in params:
        tsne_kwargs["n_iter"] = n_iter
    elif "max_iter" in params:
        tsne_kwargs["max_iter"] = n_iter
    else:
        print("Warning: TSNE implementation does not expose n_iter/max_iter; using library defaults.")
    return TSNE(**tsne_kwargs)


def _fit_tsne(
    matrix: np.ndarray,
    *,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    seed: Optional[int],
) -> np.ndarray:
    sample_count = matrix.shape[0]
    if sample_count < 2:
        raise ValueError("Need at least two samples for t-SNE")

    max_perplexity = min(perplexity, max(1.0, sample_count - 1.0))
    tsne = _instantiate_tsne(
        perplexity=max_perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        seed=seed,
    )
    return tsne.fit_transform(matrix)


def _build_output_dir(
    base_dir: Path,
    prefix: str,
    checkpoint_path: Optional[Path],
) -> Path:
    timestamp = extract_checkpoint_timestamp(checkpoint_path) if checkpoint_path is not None else None
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = base_dir / f"{prefix}_{timestamp}_tsne"
    target.mkdir(parents=True, exist_ok=True)
    return target


def main() -> None:
    projection_config_path = _determine_projection_config_path()
    proj_cfg = _load_projection_config(projection_config_path)
    base_config = tu.load_config(proj_cfg.model_config_path)

    seed = proj_cfg.seed if proj_cfg.seed is not None else base_config.seed
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

    encoder = tu.build_encoder_from_config(base_config.model).to(device)
    load_encoder_checkpoint(encoder, proj_cfg.encoder_checkpoint, device)
    encoder.eval()

    projection_head = _load_projection_head(
        encoder=encoder,
        projection_checkpoint_path=proj_cfg.projection_checkpoint,
        device=device,
    )

    module = TimeSeriesDataModule(
        dataset_name=proj_cfg.dataset_name,
        dataset_names=proj_cfg.dataset_names,
        data_dir=str(proj_cfg.data_dir),
        batch_size=proj_cfg.batch_size,
        val_batch_size=proj_cfg.val_batch_size,
        num_workers=proj_cfg.num_workers,
        pin_memory=True,
        normalize=True,
        filename=str(proj_cfg.filename) if proj_cfg.filename is not None else None,
        train=True,
        val=True,
        test=True,
    )

    dataset_groups = module.get_dataloaders()
    if not dataset_groups:
        raise RuntimeError("No datasets available for visualization")

    dataset_entries: List[Dict[str, object]] = []

    for group in dataset_groups:
        loader, _dataset_obj, split_used = select_loader(group, proj_cfg.split)
        if loader is None:
            print(f"Skipping dataset '{group.name}' because split '{proj_cfg.split}' is unavailable.")
            continue

        dataset_label = group.name
        dataset_type = _infer_dataset_type(dataset_label)

        try:
            embeddings, sample_indices = _collect_embeddings_for_dataset(
                encoder=encoder,
                projection_head=projection_head,
                loader=loader,
                device=device,
                samples_per_dataset=proj_cfg.samples_per_dataset,
                sequence_first_input=proj_cfg.sequence_first_input,
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
                "embeddings": embeddings,
                "indices": sample_indices,
            }
        )
        print(f"  Collected {sample_total} embeddings for '{dataset_label}' ({dataset_type}).")

    if not dataset_entries:
        raise RuntimeError("No embeddings collected from the requested datasets.")

    embedding_blocks = [np.asarray(entry["embeddings"], dtype=np.float32) for entry in dataset_entries]
    embedding_matrix = np.concatenate(embedding_blocks, axis=0)

    coords = _fit_tsne(
        embedding_matrix,
        perplexity=proj_cfg.perplexity,
        learning_rate=proj_cfg.learning_rate,
        n_iter=proj_cfg.n_iter,
        seed=seed,
    )

    records: List[Dict[str, object]] = []
    for entry in dataset_entries:
        embeddings = np.asarray(entry["embeddings"], dtype=np.float32)
        count = embeddings.shape[0]
        indices = entry["indices"]
        for idx in range(count):
            records.append(
                {
                    "dataset": entry["name"],
                    "dataset_type": entry["type"],
                    "split": entry["split"],
                    "sample_in_dataset": indices[idx] if idx < len(indices) else idx,
                }
            )

    coords_df = pd.DataFrame.from_records(records)
    coords_df["tsne_x"] = coords[:, 0]
    coords_df["tsne_y"] = coords[:, 1]

    output_dir = _build_output_dir(results_dir, proj_cfg.output_prefix, proj_cfg.encoder_checkpoint)
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

    ax.set_title("ICML Encoder Embeddings (t-SNE)")
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
    print(f"  TSNE config:      {projection_config_path}")
    print(f"  Encoder config:   {proj_cfg.model_config_path}")
    print(f"  Encoder checkpoint: {proj_cfg.encoder_checkpoint}")
    if proj_cfg.projection_checkpoint is not None:
        print(f"  Projection head:  {proj_cfg.projection_checkpoint}")


if __name__ == "__main__":
    main()
