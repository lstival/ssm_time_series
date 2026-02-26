from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ssm_time_series.data.dataloaders.cronos_loader_ts import load_cronos_time_series_dataset


def resolve_path(base: Path, candidate: Optional[Path | str]) -> Optional[Path]:
    """Resolve an absolute or relative path."""
    if candidate is None:
        return None
    candidate = Path(candidate).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def prepare_dataset(
    config_path: Path,
    config_data: dict,
) -> Dataset:
    """Load the dataset based on configuration."""
    cronos_config = config_data.get("cronos_config")
    if cronos_config is None:
        # Fallback to local config relative to the config file
        configs_dir = config_path.parent
        cronos_config = configs_dir / "cronos_loader_example.yaml"
    
    cronos_config = resolve_path(config_path.parent, cronos_config)
    if cronos_config is None or not cronos_config.exists():
        raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

    split = config_data.get("split")
    patch_length = config_data.get("patch_length")
    load_kwargs = dict(config_data.get("load_kwargs", {}) or {})
    normalize = config_data.get("normalize", True)

    # Set offline cache directory relative to the project root (assumed parent of scripts/configs)
    root_dir = config_path.parent.parent
    data_dir = root_dir / "data"
    load_kwargs.setdefault("offline_cache_dir", str(data_dir))
    load_kwargs.setdefault("force_offline", True)
    
    print(f"Using local data directory: {data_dir}")

    dataset = load_cronos_time_series_dataset(
        str(cronos_config),
        split=split,
        patch_length=patch_length,
        load_kwargs=load_kwargs,
        normalize=normalize,
    )
    return dataset


def split_dataset(
    dataset: Dataset,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[Dataset, Optional[Dataset]]:
    """Split dataset into train and validation subsets."""
    val_ratio = max(0.0, min(float(val_ratio), 0.9))
    if val_ratio == 0.0 or len(dataset) < 2:
        return dataset, None

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size == 0:
        train_size, val_size = len(dataset) - 1, 1

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    *,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create standard DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader
