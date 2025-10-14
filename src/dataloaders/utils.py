"""Helper utilities for discovering datasets and constructing dataset splits."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

from torch.utils.data import Dataset, Subset

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".csv", ".txt", ".npz")


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
