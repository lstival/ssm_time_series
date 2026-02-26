"""Facilities for building concatenated dataloaders across multiple datasets."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from torch.utils.data import ConcatDataset, DataLoader, Dataset
from ssm_time_series.data.data_provider.data_loader import Dataset_Custom, Dataset_Solar, Dataset_PEMS, Dataset_ETT_hour, Dataset_ETT_minute

from .utils import discover_dataset_files, split_dataset

logger = logging.getLogger(__name__)


class DatasetLoaders(NamedTuple):
    """Grouped dataloaders produced for a single dataset file."""

    name: str
    train: Optional[DataLoader]
    val: Optional[DataLoader]
    test: Optional[DataLoader]
    train_dataset: Optional[Dataset]
    val_dataset: Optional[Dataset]
    test_dataset: Optional[Dataset]


class _DatasetSplits(NamedTuple):
    """Dataset-level train/val/test splits prior to dataloader construction."""

    name: str
    train: Optional[Dataset]
    val: Optional[Dataset]
    test: Optional[Dataset]


def _try_make_dataset(
    root: str,
    data_path: str,
    flag: str,
    normalize: bool,
    sample_size: Optional[Tuple[int, int, int]],
) -> Dataset:
    """Instantiate ``Dataset_Custom`` while being tolerant to optional kwargs."""
    path = (data_path or "").lower()
    try:
        if path.endswith(".txt"):
            return Dataset_Solar(
                root_path=root,
                flag=flag,
                data_path=data_path,
                scale=normalize,
                size=sample_size,
            )

        if path.endswith(".npz"):
            return Dataset_PEMS(
                root_path=root,
                flag=flag,
                data_path=data_path,
                scale=normalize,
                size=sample_size,
            )
        if "h1" in path or "h2" in path:
            return Dataset_ETT_hour(
            root_path=root,
            flag=flag,
            data_path=data_path,
            scale=normalize,
            size=sample_size,
            )

        if "m1" in path or "m2" in path:
            return Dataset_ETT_minute(
            root_path=root,
            flag=flag,
            data_path=data_path,
            scale=normalize,
            size=sample_size,
            )
        return Dataset_Custom(
            root_path=root,
            flag=flag,
            data_path=data_path,
            scale=normalize,
            size=sample_size,
        )
        
    except TypeError:
        print("Error in Read data")
        # Backwards compatibility if ``scale`` isn't supported.
        if path.endswith(".txt"):
            return Dataset_Solar(
                root_path=root,
                flag=flag,
                data_path=data_path,
                scale=normalize,
                size=sample_size,
            )

        if path.endswith(".npz"):
            return Dataset_PEMS(
                root_path=root,
                flag=flag,
                data_path=data_path,
                scale=normalize,
                size=sample_size,
            )
        if "h1" in path or "h2" in path:
            return Dataset_ETT_hour(
            root_path=root,
            flag=flag,
            data_path=data_path,
            scale=normalize,
            size=sample_size,
            )

        if "m1" in path or "m2" in path:
            return Dataset_ETT_minute(
            root_path=root,
            flag=flag,
            data_path=data_path,
            scale=normalize,
            size=sample_size,
            )
        return Dataset_Custom(
            root_path=root,
            flag=flag,
            data_path=data_path,
            scale=normalize,
            size=sample_size,
        )


def _collect_dataset_splits(
    root_path: str,
    *,
    normalize: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    include_train: bool = True,
    include_val: bool = True,
    include_test: bool = False,
    filename: Optional[str] = None,
    dataset_files: Optional[Dict[str, str]] = None,
    sample_size: Optional[Tuple[int, int, int]] = None,
) -> Tuple[List[_DatasetSplits], Dict[str, str]]:
    """Create dataset splits for each discovered dataset prior to building loaders."""
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must equal 1.0"

    discovered_files = dataset_files or discover_dataset_files(root_path, filename=filename)
    if not discovered_files:
        raise FileNotFoundError(f"No dataset files discovered under '{root_path}'.")

    dataset_splits: List[_DatasetSplits] = []
    skipped: Dict[str, str] = {}

    for relative_path, absolute_path in sorted(discovered_files.items()):
        data_root = os.path.dirname(absolute_path) or "."
        file_name = os.path.basename(absolute_path)

        try:
            train_dataset = _try_make_dataset(
                data_root,
                file_name,
                "train",
                normalize,
                sample_size,
            )
        except Exception as exc:  # pragma: no cover - logging warning path
            skipped[relative_path] = str(exc)
            logger.warning("Skipping %s: failed to create train split (%s)", relative_path, exc)
            continue

        val_dataset: Optional[Dataset] = None
        test_dataset: Optional[Dataset] = None

        if include_val:
            try:
                val_dataset = _try_make_dataset(
                    data_root,
                    file_name,
                    "val",
                    normalize,
                    sample_size,
                )
            except Exception:
                val_dataset = None

        if include_test:
            try:
                test_dataset = _try_make_dataset(
                    data_root,
                    file_name,
                    "test",
                    normalize,
                    sample_size,
                )

            except Exception:
                test_dataset = None

        train_source: Optional[Dataset] = train_dataset if include_train else None
        val_source: Optional[Dataset] = val_dataset if include_val else None

        test_source: Optional[Dataset] = test_dataset if include_test else None

        dataset_splits.append(
            _DatasetSplits(
                name=relative_path,
                train=train_source,
                val=val_source,
                test=test_source,
            )
        )

    return dataset_splits, skipped


def _make_loader(
    dataset: Optional[Dataset],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    collate_fn: Optional[Callable],
) -> Optional[DataLoader]:
    """Create a dataloader when ``dataset`` is available and non-empty."""
    if dataset is None:
        return None

    try:
        if len(dataset) == 0:
            return None
    except TypeError:
        # Some dataset wrappers may not support ``len``; fall back to construction.
        pass

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
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
    collate_fn: Optional[Callable] = None,
    sample_size: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create dataloaders that concatenate every dataset discovered under ``root_path``."""
    dataset_splits, skipped = _collect_dataset_splits(
        root_path,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        include_train=include_train,
        include_val=include_val,
        include_test=include_test,
        filename=filename,
        dataset_files=dataset_files,
        sample_size=sample_size,
    )

    if skipped:
        logger.info("Skipped datasets due to errors: %s", skipped)

    effective_val_batch = val_batch_size or batch_size

    train_parts: List[Dataset] = [split.train for split in dataset_splits if split.train is not None]
    val_parts: List[Dataset] = [split.val for split in dataset_splits if split.val is not None]
    test_parts: List[Dataset] = [split.test for split in dataset_splits if split.test is not None]

    combined_train = ConcatDataset(train_parts) if train_parts else None
    combined_val = ConcatDataset(val_parts) if val_parts else None
    combined_test = ConcatDataset(test_parts) if test_parts else None

    train_loader = _make_loader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = _make_loader(
        combined_val,
        batch_size=effective_val_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = _make_loader(
        combined_test,
        batch_size=effective_val_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def build_dataset_loader_list(
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
    collate_fn: Optional[Callable] = None,
    sample_size: Optional[Tuple[int, int, int]] = None,
) -> List[DatasetLoaders]:
    """Build a list of dataloaders, one entry per dataset under ``root_path``."""
    dataset_splits, skipped = _collect_dataset_splits(
        root_path,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        include_train=include_train,
        include_val=include_val,
        include_test=include_test,
        filename=filename,
        dataset_files=dataset_files,
        sample_size=sample_size,
    )

    if skipped:
        logger.info("Skipped datasets due to errors: %s", skipped)

    effective_val_batch = val_batch_size or batch_size

    grouped_loaders: List[DatasetLoaders] = []
    for split in dataset_splits:
        train_loader = _make_loader(
            split.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        val_loader = _make_loader(
            split.val,
            batch_size=effective_val_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_loader = _make_loader(
            split.test,
            batch_size=effective_val_batch,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        grouped_loaders.append(
            DatasetLoaders(
                name=split.name,
                train=train_loader,
                val=val_loader,
                test=test_loader,
                train_dataset=split.train,
                val_dataset=split.val,
                test_dataset=split.test,
            )
        )

    return grouped_loaders
