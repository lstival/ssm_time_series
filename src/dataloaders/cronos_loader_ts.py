"""Chronos time-series loader that returns fixed-length torch tensors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .cronos_dataset import load_chronos_datasets, to_pandas

_DEFAULT_PATCH_LENGTH = 384
_DATASET_NAMES_KEY = "datasets_to_load"
_PATCH_LENGTH_KEY = "patch_length"
_SPLIT_KEY = "split"
_TARGET_DTYPE_KEY = "target_dtype"
_LOAD_KWARGS_KEY = "load_kwargs"
_KNOWN_GROUP_COLUMNS: Tuple[str, ...] = ("item_id", "series", "segment")


class ChronosTimeSeriesDataset(Dataset):
    """Dataset that yields patched time-series tensors and their parent ids."""

    def __init__(
        self,
        patches: torch.Tensor,
        series_ids: torch.Tensor,
        series_id_lookup: Dict[int, Any],
    ) -> None:
        if patches.shape[0] != series_ids.shape[0]:
            raise ValueError("Patch tensor and index tensor must have matching length.")
        self._patches = patches
        self._series_ids = series_ids
        self.series_id_lookup = series_id_lookup

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._patches.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._patches[idx], self._series_ids[idx]

    @property
    def patch_length(self) -> int:
        return self._patches.shape[1]

    def original_series_id(self, numeric_id: int) -> Any:
        """Resolve the original index value for a numeric series identifier."""
        return self.series_id_lookup[numeric_id]


def build_cronos_time_series_loader(
    config_path: str,
    *,
    split: Optional[str] = None,
    patch_length: Optional[int] = None,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    target_dtype: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """Construct a PyTorch dataloader over patched Chronos time-series."""

    dataset = load_cronos_time_series_dataset(
        config_path,
        split=split,
        patch_length=patch_length,
        target_dtype=target_dtype,
        load_kwargs=load_kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_cronos_time_series_dataset(
    config_path: str,
    *,
    split: Optional[str] = None,
    patch_length: Optional[int] = None,
    target_dtype: Optional[str] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
) -> ChronosTimeSeriesDataset:
    """Load Chronos datasets and convert them into a patched torch dataset."""

    config_path = str(Path(config_path).expanduser())
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    dataset_names = _find_config_value(raw_config, _DATASET_NAMES_KEY)
    if not dataset_names:
        raise ValueError(f"'{_DATASET_NAMES_KEY}' must be provided in the config: {config_path}")
    if not isinstance(dataset_names, Sequence) or isinstance(dataset_names, (str, bytes)):
        raise TypeError(f"'{_DATASET_NAMES_KEY}' must be a sequence of dataset names.")

    resolved_patch_length = _coalesce_config_integers(
        patch_length,
        _find_config_value(raw_config, _PATCH_LENGTH_KEY),
        fallback=_DEFAULT_PATCH_LENGTH,
    )

    resolved_split = _coalesce_config_strings(split, _find_config_value(raw_config, _SPLIT_KEY), fallback="train")
    resolved_target_dtype = _coalesce_config_strings(target_dtype, _find_config_value(raw_config, _TARGET_DTYPE_KEY))

    resolved_load_kwargs = dict(_find_config_value(raw_config, _LOAD_KWARGS_KEY) or {})
    if load_kwargs:
        resolved_load_kwargs.update(load_kwargs)

    hf_dataset = load_chronos_datasets(
        dataset_names,
        split=resolved_split,
        target_dtype=resolved_target_dtype,
        **resolved_load_kwargs,
    )

    dataframe = to_pandas(hf_dataset)
    patches, series_ids, id_lookup = _dataframe_to_patches(dataframe, resolved_patch_length)

    return ChronosTimeSeriesDataset(patches, series_ids, id_lookup)


def _find_config_value(config: Any, key: str) -> Any:
    if isinstance(config, dict):
        if key in config:
            return config[key]
        for value in config.values():
            found = _find_config_value(value, key)
            if found is not None:
                return found
    if isinstance(config, list):
        for item in config:
            found = _find_config_value(item, key)
            if found is not None:
                return found
    return None


def _coalesce_config_integers(*values: Optional[int], fallback: int) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(fallback)


def _coalesce_config_strings(*values: Optional[str], fallback: Optional[str] = None) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def _dataframe_to_patches(dataframe: "pd.DataFrame", patch_length: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Any]]:
    if patch_length <= 0:
        raise ValueError("patch_length must be a positive integer.")
    if "target" not in dataframe.columns:
        raise KeyError("Expected 'target' column in dataframe.")

    groups = _group_dataframe(dataframe)

    patch_chunks: List[torch.Tensor] = []
    series_index_chunks: List[torch.Tensor] = []
    index_lookup: Dict[int, Any] = {}

    current_series_idx = 0
    for series_key, series_frame in groups:
        values = series_frame["target"].to_numpy()
        patches = _series_to_patches(values, patch_length)
        patch_tensor = torch.from_numpy(patches)
        patch_chunks.append(patch_tensor)
        series_index_chunks.append(torch.full((patch_tensor.shape[0],), current_series_idx, dtype=torch.long))
        index_lookup[current_series_idx] = series_key
        current_series_idx += 1

    if not patch_chunks:
        raise ValueError("No time series found after grouping dataframe by index.")

    all_patches = torch.cat(patch_chunks, dim=0)
    all_series_ids = torch.cat(series_index_chunks, dim=0)
    return all_patches, all_series_ids, index_lookup


def _group_dataframe(dataframe: "pd.DataFrame") -> Iterable[Tuple[Any, "pd.DataFrame"]]:
    if dataframe.index.duplicated().any():
        return dataframe.groupby(level=0, sort=False)

    for column in _KNOWN_GROUP_COLUMNS:
        if column in dataframe.columns:
            return dataframe.groupby(column, sort=False)

    raise ValueError("Could not infer series groups; ensure the dataframe index contains duplicates.")


def _series_to_patches(values: np.ndarray, patch_length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, copy=False)
    length = arr.shape[0]

    if length == 0:
        return np.zeros((1, patch_length), dtype=np.float32)

    if length <= patch_length:
        padded = np.zeros((patch_length,), dtype=np.float32)
        padded[:length] = arr
        return padded.reshape(1, patch_length)

    total_patches = math.ceil(length / patch_length)
    max_start = length - patch_length
    windows: List[np.ndarray] = []

    for idx in range(total_patches):
        start = idx * patch_length
        if start > max_start:
            start = max_start
        end = start + patch_length
        windows.append(arr[start:end])

    return np.stack(windows, axis=0)


__all__ = [
    "ChronosTimeSeriesDataset",
    "build_cronos_time_series_loader",
    "load_cronos_time_series_dataset",
]
