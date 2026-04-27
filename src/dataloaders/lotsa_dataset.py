"""Load Salesforce/lotsa_data subsets from the lustre HuggingFace cache."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import datasets


LOTSA_REPO_ID = "Salesforce/lotsa_data"
CHRONOS_REPO_ID = "autogluon/chronos_datasets"
LOTSA_CACHE_DIR = "/lustre/nobackup/WUR/AIN/stiva001/hf_datasets"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cast_target_dtype(ds: datasets.Dataset, dtype: str) -> datasets.Dataset:
    if "target" not in ds.column_names:
        return ds
    target_feature = ds.features["target"]

    def _is_ok(feature) -> bool:
        if isinstance(feature, datasets.Sequence):
            inner = feature.feature
            return isinstance(inner, datasets.Value) and inner.dtype == dtype
        if isinstance(feature, datasets.Value):
            return feature.dtype == dtype
        return False

    if _is_ok(target_feature):
        return ds

    def _build(feature):
        if isinstance(feature, datasets.Sequence):
            return datasets.Sequence(feature=datasets.Value(dtype), length=feature.length)
        if isinstance(feature, datasets.Value):
            return datasets.Value(dtype)
        return feature

    try:
        return ds.cast_column("target", _build(target_feature))
    except (TypeError, ValueError):
        return ds


def _normalize_per_series(
    ds: datasets.Dataset,
    *,
    column: str = "target",
    epsilon: float = 1e-12,
) -> datasets.Dataset:
    if column not in ds.column_names:
        return ds

    feature = ds.features[column]
    if isinstance(feature, datasets.Sequence) and isinstance(feature.feature, datasets.Value):
        target_dtype = np.dtype(feature.feature.dtype)
    elif isinstance(feature, datasets.Value):
        target_dtype = np.dtype(feature.dtype)
    else:
        target_dtype = np.dtype(np.float32)

    out_dtype = target_dtype if getattr(target_dtype, "kind", "f") in ("f", "c") else np.dtype(np.float32)

    def _batch(batch):
        normalized, mins, maxs = [], [], []
        for seq in batch[column]:
            arr = np.asarray(seq, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.size == 0 or np.isnan(arr).all():
                normalized.append(arr.astype(out_dtype).tolist())
                mins.append(np.zeros(arr.shape[1:], dtype=out_dtype).tolist())
                maxs.append(np.zeros(arr.shape[1:], dtype=out_dtype).tolist())
                continue
            s_min = np.nanmin(arr, axis=0)
            s_max = np.nanmax(arr, axis=0)
            rng = s_max - s_min
            safe_rng = np.where(np.abs(rng) < epsilon, 1.0, rng)
            norm = (arr - s_min) / safe_rng
            norm = np.where(np.abs(rng) < epsilon, 0.0, norm)
            normalized.append(norm.astype(out_dtype).tolist())
            mins.append(s_min.astype(out_dtype).tolist())
            maxs.append(s_max.astype(out_dtype).tolist())
        return {column: normalized, f"{column}_min": mins, f"{column}_max": maxs}

    return ds.map(_batch, batched=True, keep_in_memory=True)


def _rename_target_column(ds: datasets.Dataset) -> datasets.Dataset:
    """Rename non-standard column names to 'target' if needed."""
    if "target" in ds.column_names:
        return ds
    for candidate in ("value", "values", "y", "series", "demand"):
        if candidate in ds.column_names:
            return ds.rename_column(candidate, "target")
    return ds


def _concatenate(
    datasets_with_names: Sequence[Tuple[str, datasets.Dataset]],
) -> Tuple[datasets.Dataset, Dict[str, str]]:
    kept: List[datasets.Dataset] = []
    failures: Dict[str, str] = {}
    reference: Optional[datasets.Dataset] = None

    for name, ds in datasets_with_names:
        if reference is None:
            kept.append(ds)
            reference = ds
            continue
        try:
            datasets.concatenate_datasets([reference, ds])
        except (TypeError, ValueError) as exc:
            failures[name] = str(exc)
            print(f"Skipping '{name}' (schema mismatch): {exc}")
            continue
        kept.append(ds)

    if not kept:
        raise ValueError("No datasets could be concatenated.")

    combined = kept[0] if len(kept) == 1 else datasets.concatenate_datasets(kept)
    if failures:
        print(f"Skipped: {', '.join(failures)}")
    return combined, failures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_lotsa_datasets(
    dataset_names: Sequence[str],
    split: str = "train",
    *,
    repo_id: str = LOTSA_REPO_ID,
    cache_dir: str = LOTSA_CACHE_DIR,
    force_offline: bool = True,
    normalize_per_series: bool = True,
    target_dtype: str = "float32",
    normalization_epsilon: float = 1e-12,
    set_numpy_format: bool = True,
    fallback_repo_id: str = CHRONOS_REPO_ID,
) -> datasets.Dataset:
    """Load and concatenate Salesforce/lotsa_data subsets.

    Parameters
    ----------
    dataset_names:
        Subset names to load (e.g. ``["m4_hourly", "ett_h1"]``).
    split:
        HuggingFace split (default ``"train"``).
    repo_id:
        HuggingFace repository identifier.
    cache_dir:
        Path to the lustre no-backup cache populated by ``download_lotsa_data.py``.
    force_offline:
        If True (default), never attempt network access — only use the local cache.
    normalize_per_series:
        Normalise each series to [0, 1] using its own min/max.
    target_dtype:
        Cast the ``target`` column to this dtype.
    normalization_epsilon:
        Minimum range value before treating a series as constant.
    set_numpy_format:
        Call ``dataset.set_format("numpy")`` on the result.

    Returns
    -------
    datasets.Dataset
        Concatenated dataset with at least a ``target`` column.
    """
    if not dataset_names:
        raise ValueError("dataset_names must not be empty.")

    # Set cache and offline flags
    prev_cache = os.environ.get("HF_DATASETS_CACHE")
    prev_offline = os.environ.get("HF_DATASETS_OFFLINE")
    prev_hub = os.environ.get("HF_HUB_OFFLINE")

    os.environ["HF_DATASETS_CACHE"] = cache_dir
    if force_offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"Offline mode — loading from: {cache_dir}")

    try:
        loaded: List[Tuple[str, datasets.Dataset]] = []
        for name in dataset_names:
            try:
                ds = datasets.load_dataset(
                    repo_id,
                    name,
                    split=split,
                    download_mode="reuse_cache_if_exists",
                )
                print(f"Loaded '{name}' from {repo_id}: {len(ds):,} rows")
            except Exception as exc:
                if fallback_repo_id and repo_id != fallback_repo_id:
                    try:
                        print(f"'{name}' not found in {repo_id}, trying {fallback_repo_id}...")
                        ds = datasets.load_dataset(
                            fallback_repo_id,
                            name,
                            split=split,
                            download_mode="reuse_cache_if_exists",
                        )
                        print(f"Loaded '{name}' from {fallback_repo_id}: {len(ds):,} rows")
                    except Exception as fallback_exc:
                        print(f"Failed to load '{name}' from both {repo_id} and {fallback_repo_id}")
                        continue
                else:
                    print(f"Failed to load '{name}': {exc}")
                    continue

            ds = _rename_target_column(ds)

            if "target" not in ds.column_names:
                print(f"Skipping '{name}': no 'target' column found.")
                continue

            if target_dtype:
                ds = _cast_target_dtype(ds, target_dtype)

            if normalize_per_series:
                ds = _normalize_per_series(
                    ds, column="target", epsilon=normalization_epsilon
                )

            # Keep only target (+ normalization stats if added)
            keep = [c for c in ("target", "target_min", "target_max") if c in ds.column_names]
            ds = ds.select_columns(keep)
            loaded.append((name, ds))

        combined, _ = _concatenate(loaded)

        if set_numpy_format:
            combined.set_format("numpy")

        return combined

    finally:
        # Restore env
        if prev_cache is not None:
            os.environ["HF_DATASETS_CACHE"] = prev_cache
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
        if force_offline:
            if prev_offline is not None:
                os.environ["HF_DATASETS_OFFLINE"] = prev_offline
            else:
                os.environ.pop("HF_DATASETS_OFFLINE", None)
            if prev_hub is not None:
                os.environ["HF_HUB_OFFLINE"] = prev_hub
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)


# ---------------------------------------------------------------------------
# Default subset list for convenience
# ---------------------------------------------------------------------------

LOTSA_DEFAULT_SUBSETS: List[str] = [
    # M-series benchmarks
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_yearly",
    "m4_weekly",
    "m4_quarterly",
    "monash_m3_monthly",
    "monash_m3_quarterly",
    "monash_m3_yearly",
    # Energy
    "solar_10min",
    "solar_weekly",
    # Transport
    "traffic_weekly",
    "taxi_30min",
    "pedestrian_counts",
    "kdd_cup_2018_with_missing",
    # Healthcare
    "hospital",
    "covid_deaths",
    # Finance
    "nn5_daily_with_missing",
    "nn5_weekly",
    # Weather & nature
    "oikolab_weather",
    "saugeenday",
    "us_births",
    "sunspot_with_missing",
    # "weather",
    # "traffic_hourly",
    # "electricity",
    # "exchange_rate"
    # NOTE: the following are ICML probe benchmarks — excluded to prevent leakage:
    #   "ett_h1", "ett_h2", "ett_m1", "ett_m2"  (ETT)
    #   "exchange_rate"                           (Exchange Rate)
    #   "traffic_hourly"                          (Traffic)
    #   "electricity_hourly"                      (Electricity)
    # weather.csv has no LOTSA equivalent and is already held out.
]


if __name__ == "__main__":
    ds = load_lotsa_datasets(["m4_daily", "ett_h1"], force_offline=False)
    print(ds)
    print(ds[0])
