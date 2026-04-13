"""DataLoader for local Monash/GluonTS Arrow datasets stored on lustre.

These datasets share the same Arrow IPC stream format (cols: id, timestamp,
target) produced by the GluonTS/Chronos download scripts.  They are safe for
CLIP pre-training because they do NOT overlap with the standard forecasting
benchmarks used for probe evaluation (ETTh1/h2, ETTm1/m2, electricity_hourly,
exchange_rate, weather, traffic).

Datasets available and their role
----------------------------------
Financial/economic (strengthens exchange_rate generalization):
  - monash_fred_md         : 107 macro-economic indicators (FRED), monthly
  - monash_car_parts       : 2 674 retail-price series, monthly

Energy (fills electricity/solar gap):
  - monash_australian_electricity: 5 demand series, 30-min
  NOTE: monash_electricity_hourly excluded — same 321-series data as electricity.csv (ICML probe).
  - electricity_15min           : 370 consumption series, 15-min
  - solar_1h                    : 1 166 solar-generation series, hourly
  - ercot                       : 8 Texas-grid series, hourly
  - wind_farms_daily            : 337 wind-farm output series, daily
  - wind_farms_hourly           : 337 wind-farm output series, hourly

Mobility/diverse:
  - uber_tlc_daily   : 262 NYC ride-share daily series
  - uber_tlc_hourly  : 262 NYC ride-share hourly series
  - mexico_city_bikes: 494 bike-share station series
  - m5               : 3 490 Walmart sales series (daily)

Usage
-----
    from dataloaders.local_dataset_loader import build_local_dataloaders

    train_dl, val_dl = build_local_dataloaders(
        dataset_names=["monash_fred_md", "monash_electricity_hourly"],
        context_length=336,
        batch_size=256,
    )

    # Or combine with LOTSA:
    from dataloaders.local_dataset_loader import load_local_datasets
    from dataloaders.lotsa_dataset import load_lotsa_datasets
    import datasets as hf_datasets

    lotsa = load_lotsa_datasets([...])
    local = load_local_datasets(["monash_fred_md", "wind_farms_daily"])
    combined = hf_datasets.concatenate_datasets([lotsa, local])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import torch
import datasets as hf_datasets
from torch.utils.data import DataLoader

from dataloaders.lotsa_loader import LotsaWindowDataset, _lotsa_collate_fn, _worker_init_fn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & catalogue
# ---------------------------------------------------------------------------

LOCAL_DATA_ROOT = Path("/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/data")

# Maps dataset name → (relative path inside LOCAL_DATA_ROOT, target column name)
_DATASET_CATALOGUE: Dict[str, Tuple[str, str]] = {
    # Financial / economic
    "monash_fred_md":              ("monash_fred_md",              "target"),
    "monash_car_parts":            ("monash_car_parts",            "target"),
    # Energy
    # NOTE: monash_electricity_hourly excluded — same 321-series UCI/GEFCOM data
    #       as electricity.csv (ICML probe benchmark). Would cause leakage.
    "monash_australian_electricity":("monash_australian_electricity","target"),
    "electricity_15min":           ("electricity_15min",           "consumption_kW"),
    "solar_1h":                    ("solar_1h",                    "power_mw"),
    "ercot":                       ("ercot",                       "target"),
    "wind_farms_daily":            ("wind_farms_daily",            "target"),
    "wind_farms_hourly":           ("wind_farms_hourly",           "target"),
    # Mobility / diverse
    "uber_tlc_daily":              ("uber_tlc_daily",              "target"),
    "uber_tlc_hourly":             ("uber_tlc_hourly",             "target"),
    "mexico_city_bikes":           ("mexico_city_bikes",           "target"),
    "m5":                          ("m5",                          "target"),
}

# Convenience groupings
FINANCIAL_DATASETS = ["monash_fred_md", "monash_car_parts"]
ENERGY_DATASETS = [
    # "monash_electricity_hourly" excluded (ICML leakage — same as electricity.csv)
    "monash_australian_electricity",
    "electricity_15min", "solar_1h", "ercot",
    "wind_farms_daily", "wind_farms_hourly",
]
MOBILITY_DATASETS = ["uber_tlc_daily", "uber_tlc_hourly", "mexico_city_bikes", "m5"]
ALL_LOCAL_DATASETS = FINANCIAL_DATASETS + ENERGY_DATASETS + MOBILITY_DATASETS


# ---------------------------------------------------------------------------
# Arrow reader
# ---------------------------------------------------------------------------

def _find_arrow_files(dataset_path: Path) -> List[Path]:
    """Recursively find all .arrow files under dataset_path."""
    return sorted(dataset_path.rglob("*.arrow"))


def _read_arrow_stream(path: Path) -> pa.Table:
    """Read a GluonTS-style Arrow IPC stream file."""
    with path.open("rb") as fh:
        reader = pa.ipc.open_stream(fh)
        return reader.read_all()


def _load_single_dataset(
    name: str,
    *,
    normalize_per_series: bool = True,
    epsilon: float = 1e-12,
) -> Optional[hf_datasets.Dataset]:
    """Load one local Arrow dataset and return a HF Dataset with a 'target' column."""
    if name not in _DATASET_CATALOGUE:
        logger.warning("'%s' not in LOCAL catalogue — skipping.", name)
        return None

    rel_path, target_col = _DATASET_CATALOGUE[name]
    dataset_path = LOCAL_DATA_ROOT / rel_path

    if not dataset_path.exists():
        logger.warning("Path not found for '%s': %s", name, dataset_path)
        return None

    arrow_files = _find_arrow_files(dataset_path)
    # Prefer the non-cache file (avoid cache-*.arrow duplicates)
    main_files = [f for f in arrow_files if not f.name.startswith("cache-")]
    if not main_files:
        main_files = arrow_files
    if not main_files:
        logger.warning("No arrow files found for '%s'.", name)
        return None

    # Read and concatenate all arrow files for this dataset
    tables = []
    for af in main_files:
        try:
            tables.append(_read_arrow_stream(af))
        except Exception as exc:
            logger.warning("Could not read %s: %s", af, exc)

    if not tables:
        return None

    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    # Normalise column name to 'target'
    if target_col != "target" and target_col in table.schema.names:
        table = table.rename_columns(
            ["target" if c == target_col else c for c in table.schema.names]
        )

    if "target" not in table.schema.names:
        logger.warning("'%s' has no usable target column — skipping.", name)
        return None

    # Convert to HF Dataset keeping only the target column (list of floats per row)
    # Each row's target is a variable-length list — cast to float32 list
    target_col_data = table.column("target")

    rows = []
    for i in range(len(table)):
        scalar = target_col_data[i]
        # Each row is a ListScalar — convert to Python list via .as_py()
        vals = scalar.as_py()
        if vals is None:
            vals = []
        # Handle nested lists (multivariate stored as list-of-lists → mean across channels)
        if vals and isinstance(vals[0], list):
            vals = np.nanmean(np.array(vals, dtype=np.float32), axis=1).tolist()
        rows.append({"target": [float(v) if v is not None else 0.0 for v in vals]})

    ds = hf_datasets.Dataset.from_list(rows)

    if normalize_per_series:
        from dataloaders.lotsa_dataset import _normalize_per_series
        ds = _normalize_per_series(ds, column="target", epsilon=epsilon)

    ds.set_format("numpy")
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_local_datasets(
    dataset_names: Sequence[str],
    *,
    normalize_per_series: bool = True,
) -> hf_datasets.Dataset:
    """Load and concatenate local Arrow datasets into a single HF Dataset.

    Parameters
    ----------
    dataset_names:
        Names from LOCAL catalogue (keys of ``_DATASET_CATALOGUE``).
    normalize_per_series:
        Apply per-series min-max normalisation (same as lotsa_dataset).

    Returns
    -------
    hf_datasets.Dataset
        Concatenated dataset with at least a ``target`` column, ready to
        be passed to ``LotsaWindowDataset``.
    """
    loaded = []
    for name in dataset_names:
        ds = _load_single_dataset(name, normalize_per_series=normalize_per_series)
        if ds is not None:
            loaded.append(ds)
            print(f"Loaded local '{name}': {len(ds):,} rows")
        else:
            print(f"Failed to load local '{name}'")

    if not loaded:
        raise ValueError(f"No local datasets could be loaded from: {list(dataset_names)}")

    if len(loaded) == 1:
        return loaded[0]

    # Align schemas before concatenating (keep only 'target' + norm stats)
    aligned = []
    for ds in loaded:
        keep = [c for c in ("target", "target_min", "target_max") if c in ds.column_names]
        aligned.append(ds.select_columns(keep))

    return hf_datasets.concatenate_datasets(aligned)


SplitSpec = Union[int, float]


def build_local_dataloaders(
    dataset_names: Optional[Sequence[str]] = None,
    *,
    context_length: int = 336,
    val_split: SplitSpec = 0.1,
    batch_size: int = 256,
    val_batch_size: Optional[int] = None,
    two_views: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    normalize_per_series: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from local Arrow datasets.

    Produces the same batch format as ``build_lotsa_dataloaders`` so the two
    can be used interchangeably or combined.

    Parameters
    ----------
    dataset_names:
        Subset of ``ALL_LOCAL_DATASETS``.  Defaults to all available.
    context_length, val_split, batch_size, two_views, num_workers, ...:
        Same semantics as ``build_lotsa_dataloaders``.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    if dataset_names is None:
        dataset_names = ALL_LOCAL_DATASETS

    merged = load_local_datasets(dataset_names, normalize_per_series=normalize_per_series)
    total = len(merged)
    logger.info("Total local series loaded: %d", total)

    test_size = val_split if isinstance(val_split, float) else val_split / total
    split = merged.train_test_split(test_size=test_size, shuffle=True, seed=seed)

    train_ds = LotsaWindowDataset(
        split["train"], context_length=context_length, two_views=two_views, seed=seed
    )
    val_ds = LotsaWindowDataset(
        split["test"], context_length=context_length, two_views=False, seed=seed + 1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_lotsa_collate_fn,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_lotsa_collate_fn,
        worker_init_fn=_worker_init_fn,
    )

    logger.info(
        "Local dataloaders ready — train: %d, val: %d, context_length: %d",
        len(train_ds), len(val_ds), context_length,
    )
    return train_loader, val_loader


def build_combined_dataloaders(
    lotsa_names: Optional[Sequence[str]] = None,
    local_names: Optional[Sequence[str]] = None,
    *,
    context_length: int = 336,
    val_split: SplitSpec = 0.1,
    batch_size: int = 256,
    val_batch_size: Optional[int] = None,
    two_views: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    normalize_per_series: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders combining LOTSA + local Arrow datasets.

    This is the recommended entry point for training with the augmented corpus.
    LOTSA datasets are loaded via ``load_lotsa_datasets``; local datasets via
    ``load_local_datasets``; both are concatenated before splitting.

    Parameters
    ----------
    lotsa_names:
        LOTSA subset names.  Defaults to ``LOTSA_DEFAULT_SUBSETS``.
    local_names:
        Local dataset names.  Defaults to ``ALL_LOCAL_DATASETS``.
    """
    from dataloaders.lotsa_dataset import load_lotsa_datasets, LOTSA_DEFAULT_SUBSETS

    if lotsa_names is None:
        lotsa_names = LOTSA_DEFAULT_SUBSETS
    if local_names is None:
        local_names = ALL_LOCAL_DATASETS

    lotsa_ds = load_lotsa_datasets(lotsa_names, normalize_per_series=normalize_per_series)
    local_ds = load_local_datasets(local_names, normalize_per_series=normalize_per_series)

    # Align schemas
    keep_cols = [c for c in ("target", "target_min", "target_max") if c in lotsa_ds.column_names]
    lotsa_ds = lotsa_ds.select_columns(keep_cols)
    local_ds = local_ds.select_columns(
        [c for c in keep_cols if c in local_ds.column_names]
    )

    merged = hf_datasets.concatenate_datasets([lotsa_ds, local_ds])
    merged.set_format("numpy")
    total = len(merged)
    logger.info("Combined corpus: %d series total", total)

    test_size = val_split if isinstance(val_split, float) else val_split / total
    split = merged.train_test_split(test_size=test_size, shuffle=True, seed=seed)

    train_ds = LotsaWindowDataset(
        split["train"], context_length=context_length, two_views=two_views, seed=seed
    )
    val_ds = LotsaWindowDataset(
        split["test"], context_length=context_length, two_views=False, seed=seed + 1
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_lotsa_collate_fn,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_lotsa_collate_fn,
        worker_init_fn=_worker_init_fn,
    )

    logger.info(
        "Combined dataloaders — train: %d, val: %d, context_length: %d",
        len(train_ds), len(val_ds), context_length,
    )
    return train_loader, val_loader


__all__ = [
    "load_local_datasets",
    "build_local_dataloaders",
    "build_combined_dataloaders",
    "ALL_LOCAL_DATASETS",
    "FINANCIAL_DATASETS",
    "ENERGY_DATASETS",
    "MOBILITY_DATASETS",
    "_DATASET_CATALOGUE",
]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Smoke test: load_local_datasets ===")
    ds = load_local_datasets(["monash_fred_md", "wind_farms_daily"])
    print(f"Combined rows: {len(ds)}")

    print("\n=== Smoke test: build_local_dataloaders ===")
    train_dl, val_dl = build_local_dataloaders(
        dataset_names=["monash_fred_md", "monash_electricity_hourly"],
        context_length=96,
        batch_size=8,
        num_workers=0,
    )
    batch = next(iter(train_dl))
    print("target shape:", batch["target"].shape)    # (8, 96, 1)
    print("lengths     :", batch["lengths"])

    print("\n=== Smoke test: build_combined_dataloaders ===")
    from dataloaders.lotsa_dataset import LOTSA_DEFAULT_SUBSETS
    train_dl, val_dl = build_combined_dataloaders(
        lotsa_names=["m4_daily", "m4_monthly"],
        local_names=["monash_fred_md", "wind_farms_daily"],
        context_length=96,
        batch_size=8,
        num_workers=0,
    )
    batch = next(iter(train_dl))
    print("combined target shape:", batch["target"].shape)
