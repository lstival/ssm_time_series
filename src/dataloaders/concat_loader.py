"""Facilities for building concatenated dataloaders across multiple datasets."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from data_provider.data_loader import Dataset_Custom

from .utils import discover_dataset_files, split_dataset

logger = logging.getLogger(__name__)


def _try_make_dataset(
    root: str,
    data_path: str,
    flag: str,
    normalize: bool,
) -> Dataset:
    """Instantiate ``Dataset_Custom`` while being tolerant to optional kwargs."""
    try:
        return Dataset_Custom(
            root_path=root,
            flag=flag,
            data_path=data_path,
            normalize=normalize,
        )
    except TypeError:
        # Backwards compatibility if ``normalize`` isn't supported.
        return Dataset_Custom(
            root_path=root,
            flag=flag,
            data_path=data_path,
        )


def build_concat_dataloaders(
    root_path: str,
    *,
    batch_size: int = 128,
    val_batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    include_train: bool = True,
    include_val: bool = True,
    include_test: bool = False,
    filename: Optional[str] = None,
    dataset_files: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create dataloaders that concatenate every dataset discovered under ``root_path``.

    Args:
        root_path: Base directory that houses dataset folders/files.
        batch_size: Batch size for the training loader.
        val_batch_size: Batch size for validation/test loaders. Defaults to ``batch_size`` when ``None``.
        num_workers: Number of workers for all dataloaders.
        pin_memory: Whether to pin memory for all dataloaders.
        normalize: Forwarded to ``Dataset_Custom`` when supported.
        train_ratio: Ratio used when a dataset lacks explicit validation split.
        val_ratio: Complementary ratio (unused when explicit validation split exists).
        include_train: Whether to build a concatenated training loader.
        include_val: Whether to build a concatenated validation loader.
        include_test: Whether to build a concatenated test loader.
        dataset_files: Optional mapping of dataset identifiers to absolute paths. When
            provided, discovery is skipped and this mapping is used instead.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``, where any missing loader is ``None``.
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must equal 1.0"

    effective_val_batch = val_batch_size or batch_size

    discovered_files = dataset_files or discover_dataset_files(root_path, filename=filename)
    if not discovered_files:
        raise FileNotFoundError(f"No dataset files discovered under '{root_path}'.")

    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []
    test_parts: List[Dataset] = []
    skipped: Dict[str, str] = {}

    for relative_path, absolute_path in discovered_files.items():
        data_root = os.path.dirname(absolute_path) or "."
        file_name = os.path.basename(absolute_path)

        try:
            train_dataset = _try_make_dataset(data_root, file_name, "train", normalize)
        except Exception as exc:  # pragma: no cover - logging warning path
            skipped[relative_path] = str(exc)
            logger.warning("Skipping %s: failed to create train split (%s)", relative_path, exc)
            continue

        val_dataset: Optional[Dataset] = None
        test_dataset: Optional[Dataset] = None

        if include_val:
            try:
                val_dataset = _try_make_dataset(data_root, file_name, "val", normalize)
                val_supported = True
            except Exception:
                val_dataset = None
                val_supported = False
        else:
            val_supported = False

        if include_test:
            try:
                test_dataset = _try_make_dataset(data_root, file_name, "test", normalize)
                test_supported = True
            except Exception:
                test_dataset = None
                test_supported = False
        else:
            test_supported = False

        if include_train:
            if include_val and not val_supported:
                train_subset, val_subset = split_dataset(train_dataset, train_ratio)
                if train_subset is not None:
                    train_parts.append(train_subset)
                if val_subset is not None:
                    val_parts.append(val_subset)
            else:
                train_parts.append(train_dataset)

        if include_val and val_dataset is not None:
            val_parts.append(val_dataset)

        if include_test and test_dataset is not None:
            test_parts.append(test_dataset)

    if skipped:
        logger.info("Skipped datasets due to errors: %s", skipped)

    combined_train = ConcatDataset(train_parts) if train_parts else None
    combined_val = ConcatDataset(val_parts) if val_parts else None
    combined_test = ConcatDataset(test_parts) if test_parts else None

    train_loader = (
        DataLoader(
            combined_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if combined_train is not None
        else None
    )

    val_loader = (
        DataLoader(
            combined_val,
            batch_size=effective_val_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if combined_val is not None
        else None
    )

    test_loader = (
        DataLoader(
            combined_test,
            batch_size=effective_val_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if combined_test is not None
        else None
    )

    return train_loader, val_loader, test_loader
