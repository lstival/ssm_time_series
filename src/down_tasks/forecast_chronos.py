"""Fine-tune a forecasting head on top of a frozen encoder using Chronos datasets."""

from __future__ import annotations

import random
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
import datasets

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

import sys

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from moco_training import resolve_path
from dataloaders.cronos_dataset import load_chronos_datasets
from util import (
    default_device,
    prepare_run_directory,
    load_encoder_checkpoint,
    simple_interpolation,
)
from down_tasks.forecast_shared import (
    apply_model_overrides,
    finalize_results,
    parse_horizon_values,
    train_dataset_group,
)


def _parse_dataset_names(raw: Optional[Iterable[str] | str]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [entry.strip() for entry in raw.split(",") if entry.strip()]
    result: List[str] = []
    for entry in raw:
        if entry is None:
            continue
        name = str(entry).strip()
        if name:
            result.append(name)
    return result


def _sequence_to_array(obj: object) -> np.ndarray:
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return arr.reshape(1, 1).astype(np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    time_dim = arr.shape[0]
    return arr.reshape(time_dim, -1).astype(np.float32)


def _split_sequences(
    sequences: Sequence[np.ndarray],
    *,
    val_ratio: float,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if not sequences or val_ratio <= 0.0:
        return list(sequences), []

    val_ratio = float(val_ratio)
    val_ratio = max(0.0, min(val_ratio, 0.9))
    indices = list(range(len(sequences)))
    if not indices:
        return [], []
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = int(len(indices) * val_ratio)
    if val_size == 0 and len(indices) > 1 and val_ratio > 0:
        val_size = 1
    if val_size >= len(indices):
        val_size = len(indices) - 1

    val_indices = set(indices[:val_size])
    train_sequences: List[np.ndarray] = []
    val_sequences: List[np.ndarray] = []
    for idx, seq in enumerate(sequences):
        if idx in val_indices:
            val_sequences.append(seq)
        else:
            train_sequences.append(seq)
    if not train_sequences and val_sequences:
        train_sequences.append(val_sequences.pop())
    return train_sequences, val_sequences


@dataclass
class ChronosDatasetGroup:
    name: str
    train: Optional[DataLoader]
    val: Optional[DataLoader]
    metadata: Dict[str, object] = field(default_factory=dict)


class ChronosForecastWindowDataset(Dataset):
    """Create sliding windows from Chronos sequences for forecasting."""

    def __init__(
        self,
        sequences: Sequence[np.ndarray],
        *,
        context_length: int,
        horizon: int,
        stride: int,
        torch_dtype: torch.dtype = torch.float32,
        max_windows_per_series: Optional[int] = None,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.dtype = torch_dtype
        self.pred_len = self.horizon
        self.series: List[np.ndarray] = []
        self.index_map: List[Tuple[int, int]] = []

        for seq in sequences:
            arr = _sequence_to_array(seq)
            total = arr.shape[0]
            required = self.context_length + self.horizon
            if total < required:
                arr = simple_interpolation(arr, required)
                total = arr.shape[0]
                # continue

            base_idx = len(self.series)
            self.series.append(arr)
            max_start = total - required
            start = 0
            windows_added = 0
            while start <= max_start:
                self.index_map.append((base_idx, start))
                windows_added += 1
                if max_windows_per_series is not None and windows_added >= max_windows_per_series:
                    break
                start += self.stride
            if windows_added == 0:
                self.index_map.append((base_idx, max_start))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        series_idx, start = self.index_map[idx]
        series = self.series[series_idx]
        ctx_end = start + self.context_length
        tgt_end = ctx_end + self.horizon
        seq_x = torch.as_tensor(series[start:ctx_end], dtype=self.dtype)
        seq_y = torch.as_tensor(series[ctx_end:tgt_end], dtype=self.dtype)
        seq_x_mark = torch.zeros_like(seq_x)
        seq_y_mark = torch.zeros_like(seq_y)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    @property
    def series_count(self) -> int:
        return len(self.series)


def _load_chronos_sequences(
    dataset_name: str,
    *,
    repo_id: str,
    split: str,
    target_dtype: Optional[str],
    normalize_per_series: bool,
    load_kwargs: Dict[str, object],
    max_series: Optional[int],
    seed: int,
) -> List[np.ndarray]:
    _ensure_hf_list_feature_registered()
    ds = load_chronos_datasets(
        [dataset_name],
        split=split,
        repo_id=repo_id,
        target_dtype=target_dtype,
        normalize_per_series=normalize_per_series,
        **load_kwargs,
    )
    sequences: List[np.ndarray] = []
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    for idx in indices:
        sample = ds[int(idx)]
        target = sample.get("target") if isinstance(sample, dict) else sample
        sequences.append(_sequence_to_array(target))
        if max_series is not None and len(sequences) >= max_series:
            break
    return sequences


def _build_dataloader(
    sequences: Sequence[np.ndarray],
    *,
    context_length: int,
    horizon: int,
    stride: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    torch_dtype: torch.dtype,
    max_windows_per_series: Optional[int],
) -> Optional[DataLoader]:
    dataset = ChronosForecastWindowDataset(
        sequences,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        torch_dtype=torch_dtype,
        max_windows_per_series=max_windows_per_series,
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_dataset_group(
    dataset_name: str,
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
) -> Optional[ChronosDatasetGroup]:
    sequences = _load_chronos_sequences(
        dataset_name,
        repo_id=repo_id,
        split=split,
        target_dtype=target_dtype,
        normalize_per_series=normalize_per_series,
        load_kwargs=load_kwargs,
        max_series=max_series,
        seed=seed,
    )
    if not sequences:
        print(f"Skipping dataset '{dataset_name}': no sequences available after loading.")
        return None

    train_sequences, val_sequences = _split_sequences(sequences, val_ratio=val_ratio, seed=seed)
    train_loader = _build_dataloader(
        train_sequences,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        torch_dtype=torch_dtype,
        max_windows_per_series=max_windows_per_series,
    )

    val_loader = _build_dataloader(
        val_sequences,
        context_length=context_length,
        horizon=horizon,
        stride=stride,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        torch_dtype=torch_dtype,
        max_windows_per_series=max_windows_per_series,
    ) if val_sequences else None

    if train_loader is None:
        print(f"Skipping dataset '{dataset_name}': no training windows after preprocessing.")
        return None

    val_sample = next(iter(val_loader))
    print(f"Val loader sample shape: {val_sample[0].shape}")

    metadata = {
        "train_series": len(train_sequences),
        "val_series": len(val_sequences),
        "train_windows": len(train_loader.dataset),  # type: ignore[attr-defined]
        "val_windows": len(val_loader.dataset) if val_loader is not None else 0,  # type: ignore[attr-defined]
        "context_length": context_length,
        "horizon": horizon,
        "stride": stride,
    }
    print(
        f"Prepared dataset '{dataset_name}': {metadata['train_series']} train series -> "
        f"{metadata['train_windows']} train windows (context={context_length}, horizon={horizon}, stride={stride})."
    )
    if val_loader is None:
        print(f"  Validation split disabled or empty.")
    else:
        print(
            f"  Validation: {metadata['val_series']} series -> {metadata['val_windows']} windows."
        )

    return ChronosDatasetGroup(
        name=dataset_name,
        train=train_loader,
        val=val_loader,
        metadata=metadata,
    )


DEFAULT_FORECAST_CONFIG = SRC_DIR / "configs" / "chronos_forecast.yaml"


def _load_yaml_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must define a mapping: {path}")
    return payload


def _resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    if candidate is None:
        return None
    candidate_path = Path(str(candidate))
    resolved = resolve_path(base, candidate_path)
    if resolved is not None:
        return resolved
    return candidate_path.expanduser().resolve()


def _coerce_path(base: Path, candidate: object, *, must_exist: bool = False, description: str) -> Path:
    resolved = _resolve_optional_path(base, candidate)
    if resolved is None:
        raise FileNotFoundError(f"{description} not found: {candidate}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _normalize_horizons(raw: object) -> List[int]:
    if raw is None:
        raise ValueError("Configuration must supply 'training.horizons'.")
    if isinstance(raw, str):
        return parse_horizon_values(raw)
    try:
        values = [int(value) for value in raw]  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("'training.horizons' must be a list of integers or a comma-separated string.") from exc
    return sorted(set(values))


_HF_LIST_FEATURE_PATCHED = True


def _ensure_hf_list_feature_registered() -> None:
    global _HF_LIST_FEATURE_PATCHED
    if _HF_LIST_FEATURE_PATCHED:
        return
    feature_registry = getattr(datasets.features, "_FEATURE_TYPES", None)
    sequence_cls = getattr(datasets.features, "Sequence", None)
    if isinstance(feature_registry, dict) and "List" not in feature_registry and sequence_cls is not None:
        feature_registry["List"] = sequence_cls
        _HF_LIST_FEATURE_PATCHED = True


if __name__ == "__main__":
    env_override = os.getenv("CHRONOS_FORECAST_CONFIG")
    config_candidate: Path | str = env_override if env_override else DEFAULT_FORECAST_CONFIG
    config_path = _resolve_optional_path(Path.cwd(), config_candidate)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Chronos forecast configuration not found: {config_candidate}")
    print(f"Using forecast configuration: {config_path}")

    forecast_cfg = _load_yaml_config(config_path)

    model_section = dict(forecast_cfg.get("model") or {})
    model_config_candidate = model_section.get("config")
    if model_config_candidate is None:
        raise ValueError("Configuration missing required key 'model.config'.")
    model_config_path = _coerce_path(
        config_path.parent,
        model_config_candidate,
        must_exist=True,
        description="Model configuration",
    )
    base_config = tu.load_config(model_config_path)

    data_cfg = dict(base_config.data or {})
    logging_cfg = dict(base_config.logging or {})

    overrides = dict(model_section.get("overrides") or {})

    training_section = dict(forecast_cfg.get("training") or {})
    horizons = _normalize_horizons(training_section.get("horizons"))
    max_horizon = max(horizons)
    print(f"Training on horizons: {horizons} (max horizon: {max_horizon})")

    seed = int(training_section.get("seed", base_config.seed))
    tu.set_seed(seed)
    torch.manual_seed(seed)

    batch_size = int(training_section.get("batch_size", data_cfg.get("batch_size", 16)))
    val_batch_size = int(training_section.get("val_batch_size", batch_size))
    epochs = int(training_section.get("epochs", 2))
    lr = float(training_section.get("lr", 3e-4))
    weight_decay = float(training_section.get("weight_decay", 1e-2))
    mlp_hidden_dim = int(training_section.get("mlp_hidden_dim", 512))
    num_workers = int(training_section.get("num_workers", data_cfg.get("num_workers", 0)))
    pin_memory = bool(training_section.get("pin_memory", data_cfg.get("pin_memory", True)))

    chronos_section = dict(forecast_cfg.get("chronos") or {})
    loader_config_candidate = chronos_section.get("config")
    chronos_cfg: Dict[str, object] = {}
    if loader_config_candidate is not None:
        loader_config_path = _coerce_path(
            config_path.parent,
            loader_config_candidate,
            must_exist=True,
            description="Chronos loader configuration",
        )
        with loader_config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Chronos loader configuration must be a mapping: {loader_config_path}")
        chronos_cfg.update(payload)

    datasets_value = chronos_section.get("datasets")
    if datasets_value is None:
        datasets_value = chronos_cfg.get("datasets_to_load") or chronos_cfg.get("datasets")
    datasets_to_load = _parse_dataset_names(datasets_value)
    if not datasets_to_load:
        raise ValueError("Chronos configuration must provide at least one dataset name.")

    split = str(chronos_section.get("split", chronos_cfg.get("split", "train")))
    repo_id = str(chronos_section.get("repo_id", chronos_cfg.get("repo_id", "autogluon/chronos_datasets")))
    target_dtype = chronos_section.get("target_dtype", chronos_cfg.get("target_dtype"))

    normalize_per_series = bool(chronos_section.get("normalize", chronos_cfg.get("normalize", True)))
    val_ratio = float(chronos_section.get("val_split", chronos_cfg.get("val_split", 0.2)))

    context_length = chronos_section.get("context_length")
    if context_length is None:
        context_length = chronos_cfg.get("context_length", chronos_cfg.get("patch_length", 384))
    context_length = int(context_length)

    stride_value = chronos_section.get("window_stride", chronos_cfg.get("window_stride"))
    stride = int(stride_value) if stride_value is not None else max_horizon
    stride = max(1, stride)

    max_windows_per_series = chronos_section.get("max_windows_per_series", chronos_cfg.get("max_windows_per_series"))
    if max_windows_per_series is not None:
        max_windows_per_series = int(max_windows_per_series)

    max_series = chronos_section.get("max_series", chronos_cfg.get("max_series"))
    if max_series is not None:
        max_series = int(max_series)

    load_kwargs: Dict[str, object] = dict(chronos_cfg.get("load_kwargs", {}) or {})
    section_load_kwargs = chronos_section.get("load_kwargs")
    if isinstance(section_load_kwargs, dict):
        load_kwargs.update(section_load_kwargs)

    force_offline = chronos_section.get("force_offline")
    if force_offline is not None:
        load_kwargs["force_offline"] = bool(force_offline)

    offline_cache_dir = load_kwargs.get("offline_cache_dir")
    if offline_cache_dir is not None:
        load_kwargs["offline_cache_dir"] = str(
            _resolve_optional_path(config_path.parent, offline_cache_dir)
        )
    if "offline_cache_dir" not in load_kwargs:
        load_kwargs["offline_cache_dir"] = str((ROOT_DIR / "data").resolve())
    print(f"Using offline cache directory: {load_kwargs['offline_cache_dir']}")
    if load_kwargs.get("force_offline"):
        print("Forcing offline mode - no network access will be attempted")

    paths_section = dict(forecast_cfg.get("paths") or {})
    encoder_checkpoint_path = _coerce_path(
        config_path.parent,
        paths_section.get(
            "encoder_checkpoint",
            ROOT_DIR / "checkpoints" / "ts_encoder_20251101_1100" / "time_series_best.pt",
        ),
        must_exist=True,
        description="Encoder checkpoint",
    )
    checkpoint_dir = _coerce_path(
        config_path.parent,
        paths_section.get("checkpoint_dir", logging_cfg.get("checkpoint_dir", ROOT_DIR / "checkpoints")),
        must_exist=False,
        description="Checkpoint directory",
    )
    results_dir = _coerce_path(
        config_path.parent,
        paths_section.get("results_dir", ROOT_DIR / "results"),
        must_exist=False,
        description="Results directory",
    )

    device = default_device()
    print(f"Using device: {device}")

    model_cfg = apply_model_overrides(
        base_config.model,
        token_size=overrides.get("token_size"),
        model_dim=overrides.get("model_dim"),
        embedding_dim=overrides.get("embedding_dim"),
        depth=overrides.get("depth"),
    )
    encoder = tu.build_encoder_from_config(model_cfg).to(device)

    if context_length < getattr(encoder, "input_dim", 1):
        raise ValueError(
            f"context_length {context_length} must be >= encoder token size {getattr(encoder, 'input_dim', '?')}."
        )

    print(f"Loading encoder checkpoint: {encoder_checkpoint_path}")
    load_encoder_checkpoint(encoder, encoder_checkpoint_path, device)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    run_root = prepare_run_directory(checkpoint_dir, "multi_horizon_forecast_chronos")
    print(f"Checkpoint root directory: {run_root}")
    print(f"Results directory: {results_dir}")

    torch_dtype = torch.float32

    dataset_groups: List[ChronosDatasetGroup] = []
    for dataset_name in datasets_to_load:
        group = build_dataset_group(
            dataset_name,
            repo_id=repo_id,
            split=split,
            target_dtype=target_dtype,
            normalize_per_series=normalize_per_series,
            load_kwargs=load_kwargs,
            context_length=context_length,
            horizon=max_horizon,
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
        if group is not None:
            dataset_groups.append(group)

    if not dataset_groups:
        print("No datasets were prepared. Check Chronos configuration or dataset availability.")

    dataset_records: List[Dict[str, object]] = []

    for group in dataset_groups:
        record = train_dataset_group(
            group_name=group.name,
            train_loader=group.train,
            val_loader=group.val,
            encoder=encoder,
            device=device,
            horizons=horizons,
            mlp_hidden_dim=mlp_hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            run_root=run_root,
            max_horizon=max_horizon,
            criterion=criterion,
        )
        if record is not None:
            dataset_records.append(record)

    if not dataset_records:
        print("No datasets were trained. Check dataset filters or availability.")

    finalize_results(
        dataset_records=dataset_records,
        horizons=horizons,
        run_root=run_root,
        results_dir=results_dir,
        filename_prefix="forecast_results_chronos",
        checkpoint_path=encoder_checkpoint_path,
        encoder_embedding_dim=getattr(encoder, "embedding_dim", None),
        mlp_hidden_dim=mlp_hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        extra_model_config={
            "context_length": context_length,
            "stride": stride,
            "split": split,
        },
    )
