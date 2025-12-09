"""Helper utilities for discovering datasets and constructing dataset splits."""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
import datasets

from dataloaders.cronos_dataset import load_chronos_datasets

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".csv", ".txt", ".npz")


def _simple_interpolation(time_series: object, target_size: int) -> object:
    """Lightweight interpolation helper to avoid importing util and creating cycles."""
    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")

    is_torch = isinstance(time_series, torch.Tensor)
    if is_torch:
        device = time_series.device
        dtype = time_series.dtype
        arr = time_series.detach().cpu().numpy()
    else:
        arr = np.asarray(time_series)

    if arr.ndim != 2:
        raise ValueError("time_series must be 2D with shape (time_steps, variables)")

    time_steps, variables = arr.shape
    if time_steps == target_size:
        return time_series.clone() if is_torch else arr.copy()

    if time_steps == 1:
        out = np.repeat(arr, target_size, axis=0)
    else:
        old_pos = np.linspace(0.0, 1.0, time_steps)
        new_pos = np.linspace(0.0, 1.0, target_size)
        out = np.empty((target_size, variables), dtype=arr.dtype)
        for idx in range(variables):
            out[:, idx] = np.interp(new_pos, old_pos, arr[:, idx])

    if is_torch:
        return torch.from_numpy(out).to(device=device, dtype=dtype)
    return out


def discover_dataset_files(
    root_path: str,
    extensions: Optional[Sequence[str]] = None,
    filename: Optional[str] = None,
) -> Dict[str, str]:
    """Recursively discover dataset files under ``root_path``.

    Args:
        root_path: Directory that contains dataset folders/files.
        extensions: Optional iterable of file extensions (case-insensitive) to include.
        filename: Optional filename or relative path (relative to ``root_path``).
            When provided, the function will return at most one matching entry (the
            first match found). Matching is case-insensitive and will match either the
            base filename or the relative path.

    Returns:
        Mapping from relative dataset path (relative to ``root_path``) to absolute path.
        If ``filename`` is provided, the mapping will contain at most one entry.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    normalized_exts = tuple(ext.lower() for ext in extensions)
    dataset_files: Dict[str, str] = {}

    filename_norm = None
    if filename is not None:
        # Normalize and lowercase for case-insensitive comparison
        filename_norm = os.path.normpath(filename).lower()

        # If filename refers to an explicit file (absolute or relative to root_path),
        # return just that file if it exists and has an allowed extension.
        candidate = filename
        if not os.path.isabs(candidate):
            candidate = os.path.join(root_path, candidate)
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate) and candidate.lower().endswith(normalized_exts):
            abs_path = os.path.abspath(candidate)
            key = os.path.relpath(abs_path, root_path)
            if key == ".":
                key = os.path.basename(abs_path)
            dataset_files[key] = abs_path
            return dataset_files

        # If explicit path not found, fall back to searching for a matching filename
        filename_basename = os.path.basename(filename_norm)
    else:
        filename_basename = None

    for current_root, _, files in os.walk(root_path):
        for file_name in files:
            if file_name.lower().endswith(normalized_exts):
                absolute_path = os.path.join(current_root, file_name)
                relative_root = os.path.relpath(current_root, root_path)
                key = os.path.join(relative_root, file_name) if relative_root != "." else file_name

                # If a filename filter is provided, check for a case-insensitive match
                if filename is not None:
                    if file_name.lower() == filename_basename or os.path.normpath(key).lower() == filename_norm:
                        dataset_files[key] = absolute_path
                        return dataset_files  # return the first match immediately
                    # otherwise skip adding this file
                    continue

                dataset_files[key] = absolute_path

    return dataset_files


def split_dataset(dataset: Dataset, train_ratio: float) -> Tuple[Optional[Subset], Optional[Subset]]:
    """Split a dataset into train/validation subsets according to ``train_ratio``.

    Args:
        dataset: Dataset to split.
        train_ratio: Value in ``(0, 1]`` representing the proportion for the train split.

    Returns:
        A tuple ``(train_subset, val_subset)``; either element may be ``None`` when empty.
    """
    total = len(dataset)
    if total == 0:
        return None, None

    split_idx = int(total * train_ratio)
    split_idx = max(0, min(split_idx, total))  # clamp to valid range

    train_subset: Optional[Subset] = None
    val_subset: Optional[Subset] = None

    if split_idx > 0:
        train_subset = Subset(dataset, range(0, split_idx))
    if split_idx < total:
        val_subset = Subset(dataset, range(split_idx, total))

    return train_subset, val_subset


@dataclass
class ChronosDatasetGroup:
    """Group containing train and validation dataloaders for a dataset."""
    name: str
    train: Optional[DataLoader]
    val: Optional[DataLoader]
    metadata: Dict[str, object] = field(default_factory=dict)


def _ensure_hf_list_feature_registered() -> None:
    """Ensure HuggingFace List feature is registered."""
    try:
        from datasets.features import features as hf_features  # type: ignore
    except Exception:
        return

    feature_registry = getattr(hf_features, "_FEATURE_TYPES", None)
    sequence_cls = getattr(hf_features, "Sequence", None)
    if not isinstance(feature_registry, dict) or sequence_cls is None:
        return

    # HF 1.13 serialized some datasets with the deprecated ``List`` feature token.
    # Modern releases removed it, so we alias it to ``Sequence`` before parsing
    # dataset metadata to keep offline caches readable.
    if "List" not in feature_registry:
        feature_registry["List"] = sequence_cls
    if getattr(hf_features, "List", None) is None:
        setattr(hf_features, "List", sequence_cls)


def _sequence_to_array(obj: object) -> np.ndarray:
    """Convert sequence object to numpy array."""
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
    """Split sequences into training and validation sets."""
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
    """Load sequences from a Chronos dataset."""
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
                arr = _simple_interpolation(arr, required)
                total = arr.shape[0]

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
    """Build a dataloader from sequences."""
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


def _build_aggregated_loader(
    dataset_groups: Sequence[ChronosDatasetGroup],
    *,
    is_train: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> Optional[DataLoader]:
    """Aggregate datasets from multiple ChronosDatasetGroup objects into one loader."""
    datasets_to_concat: List[Dataset] = []
    collate_fn = None

    for group in dataset_groups:
        loader = group.train if is_train else group.val
        if loader is None:
            continue
        ds = getattr(loader, "dataset", None)
        if ds is None or len(ds) == 0:  # type: ignore[arg-type]
            continue
        datasets_to_concat.append(ds)
        if collate_fn is None and loader.collate_fn is not None:
            collate_fn = loader.collate_fn

    if not datasets_to_concat:
        return None

    merged_dataset: Dataset
    if len(datasets_to_concat) == 1:
        merged_dataset = datasets_to_concat[0]
    else:
        merged_dataset = ConcatDataset(datasets_to_concat)

    return DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
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
    """Build a dataset group with train and validation loaders."""
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

    if val_loader is not None and len(val_loader.dataset) > 0:  # type: ignore[attr-defined]
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
    if val_loader is None or len(val_loader.dataset) == 0:  # type: ignore[attr-defined]
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
