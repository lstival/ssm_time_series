"""Shared helpers for encoder embedding projection scripts."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
import yaml
from torch.utils.data import DataLoader, Dataset

from evaluation_down_tasks.zeroshot_utils import extract_checkpoint_timestamp
from dataloaders.cronos_dataset import load_chronos_datasets
from dataloaders.utils import _ensure_hf_list_feature_registered
from util import build_projection_head


@dataclass
class ProjectionConfig:
    config_path: Path
    model_config_path: Path
    encoder_checkpoint: Path
    projection_checkpoint: Optional[Path]
    visual_encoder_checkpoint: Optional[Path]
    visual_projection_checkpoint: Optional[Path]
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


DEFAULT_CHRONOS_DATASETS: List[str] = [
    "m4_daily",
    "electricity_15min",
    "solar_1h",
    "taxi_30min",
    "monash_hospital",
]
MAX_CHRONOS_DATASETS = 5
CHRONOS_REPO_ID = "autogluon/chronos_datasets"


@dataclass
class ChronosDatasetGroup:
    """Canonical Chronos dataset packaging for projection scripts."""

    name: str
    loader: DataLoader
    split: str


class ChronosEmbeddingDataset(Dataset):
    """Simple dataset wrapping Chronos time-series sequences."""

    def __init__(self, sequences: Sequence[np.ndarray]):
        self.sequences = [np.asarray(seq, dtype=np.float32) for seq in sequences if len(seq) > 0]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        arr = self.sequences[index]
        tensor = torch.from_numpy(arr).to(dtype=torch.float32)
        return tensor.unsqueeze(-1)


def chronos_embedding_collate(batch: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    if not batch:
        raise ValueError("Chronos collate received an empty batch.")
    lengths = [sample.size(0) for sample in batch]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, 1, dtype=torch.float32)
    for idx, sample in enumerate(batch):
        length = lengths[idx]
        flat = sample.view(-1)
        padded[idx, :length, 0] = flat[:length]
    return (padded,)


def resolve_chronos_dataset_names(
    *,
    dataset_names: Optional[List[str]],
    dataset_name: str,
    fallback: Optional[Sequence[str]] = None,
) -> List[str]:
    ordered: List[str] = []
    if dataset_names:
        ordered.extend(str(name) for name in dataset_names)
    elif dataset_name:
        ordered.append(str(dataset_name))
    if not ordered:
        fallback_list = list(fallback or DEFAULT_CHRONOS_DATASETS)
    else:
        fallback_list = []

    seen: Dict[str, None] = {}
    result: List[str] = []
    for name in ordered or fallback_list:
        key = str(name).strip()
        if not key:
            continue
        if "." in key:
            key = Path(key).stem
        if key and key not in seen:
            seen[key] = None
            result.append(key)
        if len(result) >= MAX_CHRONOS_DATASETS:
            break
    if not result:
        result = list(DEFAULT_CHRONOS_DATASETS[:MAX_CHRONOS_DATASETS])
    return result[:MAX_CHRONOS_DATASETS]


def load_chronos_sequences(dataset_name: str, *, limit: int) -> List[np.ndarray]:
    _ensure_hf_list_feature_registered()
    try:
        hf_dataset = load_chronos_datasets(
            [dataset_name],
            split="train",
            repo_id=CHRONOS_REPO_ID,
            target_dtype="float32",
            normalize_per_series=True,
            force_offline=False,
        )
    except (ValueError, KeyError) as exc:
        print(f"Skipping Chronos dataset '{dataset_name}' due to load error: {exc}")
        return []
    total = len(hf_dataset)
    if total == 0:
        return []

    max_items = total if limit <= 0 else min(total, limit)
    sequences: List[np.ndarray] = []
    for idx in range(max_items):
        sample = hf_dataset[int(idx)]
        target = sample.get("target") if isinstance(sample, dict) else sample
        array = np.asarray(target, dtype=np.float32)
        if array.size == 0:
            continue
        sequences.append(array)
    return sequences


def build_chronos_dataset_groups(*, proj_cfg) -> List[ChronosDatasetGroup]:
    dataset_names = resolve_chronos_dataset_names(
        dataset_names=proj_cfg.dataset_names,
        dataset_name=proj_cfg.dataset_name,
    )

    groups: List[ChronosDatasetGroup] = []
    for dataset_name in dataset_names:
        sequences = load_chronos_sequences(dataset_name, limit=proj_cfg.samples_per_dataset)
        if not sequences:
            print(f"Skipping Chronos dataset '{dataset_name}' due to missing sequences.")
            continue
        dataset = ChronosEmbeddingDataset(sequences)
        loader = DataLoader(
            dataset,
            batch_size=proj_cfg.batch_size,
            shuffle=False,
            num_workers=proj_cfg.num_workers,
            collate_fn=chronos_embedding_collate,
        )
        groups.append(ChronosDatasetGroup(name=dataset_name, loader=loader, split="train"))
    return groups


def determine_config_path(*, env_var: str, default_path: Path) -> Path:
    env_value = os.environ.get(env_var)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return candidate
    resolved_default = default_path.expanduser().resolve()
    if resolved_default.exists():
        return resolved_default
    raise FileNotFoundError(
        f"Projection config not found. Provide path via {env_var} or create {default_path}."
    )


def resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    if candidate is None:
        return None
    path_candidate = Path(str(candidate)).expanduser()
    if not path_candidate.is_absolute():
        path_candidate = (base / path_candidate).resolve()
    else:
        path_candidate = path_candidate.resolve()
    return path_candidate


def resolve_required_path(base: Path, candidate: Optional[object], *, description: str) -> Path:
    path = resolve_optional_path(base, candidate)
    if path is None:
        raise ValueError(f"Missing required path for {description}")
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_projection_config(
    config_path: Path,
    *,
    default_model_config_path: Path,
    default_output_dir: Path,
    default_output_prefix: str,
    default_data_dir: Path,
    default_split: str,
) -> ProjectionConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    base_dir = config_path.parent
    paths_section = dict(payload.get("paths") or {})
    tsne_section = dict(payload.get("tsne") or {})
    data_section = dict(payload.get("data") or {})

    model_config_path = resolve_optional_path(base_dir, paths_section.get("model_config"))
    if model_config_path is None:
        model_config_path = default_model_config_path
    model_config_path = model_config_path.expanduser().resolve()
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    encoder_checkpoint = resolve_required_path(
        base_dir,
        paths_section.get("encoder_checkpoint"),
        description="encoder checkpoint",
    )
    projection_checkpoint = resolve_optional_path(base_dir, paths_section.get("projection_checkpoint"))
    visual_encoder_checkpoint = resolve_optional_path(base_dir, paths_section.get("visual_encoder_checkpoint"))
    visual_projection_checkpoint = resolve_optional_path(base_dir, paths_section.get("visual_projection_checkpoint"))
    output_dir = resolve_optional_path(base_dir, paths_section.get("output_dir"))
    if output_dir is None:
        output_dir = default_output_dir.expanduser().resolve()

    data_dir_candidate = data_section.get("data_dir")
    if data_dir_candidate is None:
        data_dir_candidate = default_data_dir
    data_dir = resolve_required_path(base_dir, data_dir_candidate, description="data directory")

    dataset_names_value = data_section.get("dataset_names")
    if isinstance(dataset_names_value, list):
        dataset_names = [str(name) for name in dataset_names_value]
    elif isinstance(dataset_names_value, str) and dataset_names_value.strip():
        dataset_names = [dataset_names_value.strip()]
    else:
        dataset_names = None

    dataset_name = str(data_section.get("dataset_name") or "").strip()
    filename = resolve_optional_path(base_dir, data_section.get("filename"))

    batch_size = int(data_section.get("batch_size", 512))
    val_batch_size = int(data_section.get("val_batch_size", batch_size))
    num_workers = int(data_section.get("num_workers", 4))
    split = str(data_section.get("split", default_split)).lower()

    output_prefix = str(tsne_section.get("output_prefix", default_output_prefix))
    sequence_first = bool(tsne_section.get("sequence_first_input", False))

    seed_value = tsne_section.get("seed")
    seed_int = int(seed_value) if seed_value is not None else None

    return ProjectionConfig(
        config_path=config_path,
        model_config_path=model_config_path,
        encoder_checkpoint=encoder_checkpoint,
        projection_checkpoint=projection_checkpoint,
        visual_encoder_checkpoint=visual_encoder_checkpoint,
        visual_projection_checkpoint=visual_projection_checkpoint,
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


def load_projection_head(
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


def infer_dataset_type(name: str) -> str:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized:
        return "dataset"
    parts = normalized.split("/")
    base = parts[0] if len(parts) > 1 else normalized
    base_path = Path(base)
    stem = base_path.stem if base_path.suffix else base_path.name
    return stem or "dataset"


def collect_embeddings_for_dataset(
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


def instantiate_tsne(
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


def fit_tsne(
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
    tsne = instantiate_tsne(
        perplexity=max_perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        seed=seed,
    )
    return tsne.fit_transform(matrix)


def build_output_dir(
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
