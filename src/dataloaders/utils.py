"""Helper utilities for discovering datasets and constructing dataset splits."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

from torch.utils.data import Dataset, Subset

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".csv", ".txt", ".npz")


def discover_dataset_files(root_path: str, extensions: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Recursively discover dataset files under ``root_path``.

    Args:
        root_path: Directory that contains dataset folders/files.
        extensions: Optional iterable of file extensions (case-insensitive) to include.

    Returns:
        Mapping from relative dataset path (relative to ``root_path``) to absolute path.
    """
    if extensions is None:
        extensions = SUPPORTED_EXTENSIONS

    normalized_exts = tuple(ext.lower() for ext in extensions)
    dataset_files: Dict[str, str] = {}

    for current_root, _, files in os.walk(root_path):
        for file_name in files:
            if file_name.lower().endswith(normalized_exts):
                absolute_path = os.path.join(current_root, file_name)
                relative_root = os.path.relpath(current_root, root_path)
                key = os.path.join(relative_root, file_name) if relative_root != "." else file_name
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
