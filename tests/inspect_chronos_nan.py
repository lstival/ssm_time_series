"""Utility script to scan Chronos datasets for NaN/Inf values using the forecast config."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

# Removed legacy sys.path hack

from ssm_time_series.models.utils import load_chronos_forecast_config
from ssm_time_series.data.dataloaders.cronos_dataset import load_chronos_datasets


# Hard-coded parameters for quick inspection. Edit these values as needed.
# - `datasets`: None uses config.datasets_to_load; otherwise provide a list of names.
# - `seed`: None uses config.seed; otherwise integer seed for shuffling series.
# - `max_series`: limit number of series per dataset (None to use config.max_series).
# - `show_examples`: how many problematic series to print per dataset.
# - `skip_normalization`: set True to disable per-series normalization when loading.
HARDCODED_PARAMS: Dict[str, Any] = {
    "datasets": None,
    "seed": None,
    "max_series": None,
    "show_examples": 5,
    "skip_normalization": False,
}


def _flatten_series(sequence: Any) -> np.ndarray:
    arr = np.asarray(sequence)
    if arr.dtype == object:
        arr = np.array(sequence, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float64, copy=False)


def _summarize_lengths(lengths: Sequence[int]) -> Dict[str, float]:
    if not lengths:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    length_min = float(min(lengths))
    length_max = float(max(lengths))
    length_mean = float(sum(lengths)) / float(len(lengths))
    return {"min": length_min, "max": length_max, "mean": length_mean}


def _format_problem_examples(records: Sequence[Dict[str, int]], limit: int) -> str:
    if not records:
        return ""
    keep = records[:limit]
    formatted = [
        f"order={entry['order']} hf_index={entry['hf_index']} count={entry['count']}"
        for entry in keep
    ]
    extras = "" if len(records) <= limit else f" (+{len(records) - limit} more)"
    return "; ".join(formatted) + extras


def inspect_dataset(
    dataset_name: str,
    *,
    config,
    seed: int,
    max_series: int | None,
    show_examples: int,
    normalize: bool,
) -> Dict[str, Any]:
    load_kwargs = dict(config.load_kwargs)
    force_offline = bool(load_kwargs.pop("force_offline", True))
    offline_cache_dir = load_kwargs.pop("offline_cache_dir", None)

    dataset = load_chronos_datasets(
        [dataset_name],
        split=config.split,
        repo_id=config.repo_id,
        target_dtype=config.target_dtype,
        normalize_per_series=normalize,
        force_offline=force_offline,
        offline_cache_dir=offline_cache_dir,
        **load_kwargs,
    )

    total_available = len(dataset)
    indices = list(range(total_available))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if max_series is not None:
        indices = indices[:max_series]

    inspected = len(indices)
    lengths: List[int] = []
    nan_records: List[Dict[str, int]] = []
    inf_records: List[Dict[str, int]] = []
    nan_values = 0
    inf_values = 0

    for order_idx, hf_idx in enumerate(indices):
        sample = dataset[int(hf_idx)]
        target = sample["target"] if isinstance(sample, dict) else sample
        series = _flatten_series(target)
        lengths.append(int(series.shape[0]))

        flat = series.reshape(-1)
        nan_mask = np.isnan(flat)
        inf_mask = np.isinf(flat)

        nan_count = int(np.count_nonzero(nan_mask))
        inf_count = int(np.count_nonzero(inf_mask))

        if nan_count:
            nan_records.append({"order": order_idx, "hf_index": int(hf_idx), "count": nan_count})
            nan_values += nan_count
        if inf_count:
            inf_records.append({"order": order_idx, "hf_index": int(hf_idx), "count": inf_count})
            inf_values += inf_count

    length_stats = _summarize_lengths(lengths)

    print(f"\nDataset: {dataset_name}")
    print(f"  available series: {total_available}")
    print(f"  inspected series: {inspected} (max_series={max_series})")
    print(
        "  sequence length stats: min={min(lengths) if lengths else 0}, "
        f"max={max(lengths) if lengths else 0}, mean={length_stats['mean']:.2f}"
    )

    if nan_records:
        percent = (len(nan_records) / inspected * 100.0) if inspected else 0.0
        print(f"  NaN series: {len(nan_records)} / {inspected} ({percent:.2f}%) | values={nan_values}")
        print(f"    examples: {_format_problem_examples(nan_records, show_examples)}")
    else:
        print("  NaN series: none detected")

    if inf_records:
        percent = (len(inf_records) / inspected * 100.0) if inspected else 0.0
        print(f"  Inf series: {len(inf_records)} / {inspected} ({percent:.2f}%) | values={inf_values}")
        print(f"    examples: {_format_problem_examples(inf_records, show_examples)}")
    else:
        print("  Inf series: none detected")

    return {
        "dataset": dataset_name,
        "available_series": total_available,
        "inspected_series": inspected,
        "length_min": length_stats["min"],
        "length_max": length_stats["max"],
        "length_mean": length_stats["mean"],
        "nan_series": len(nan_records),
        "nan_values": nan_values,
        "inf_series": len(inf_records),
        "inf_values": inf_values,
    }


def main() -> None:
    cli_settings = dict(HARDCODED_PARAMS)
    config = load_chronos_forecast_config()

    # Apply hard-coded parameters (edit `HARDCODED_PARAMS` above to change behavior)
    dataset_names: Iterable[str] = (
        HARDCODED_PARAMS["datasets"] if HARDCODED_PARAMS["datasets"] else config.datasets_to_load
    )
    seed = int(HARDCODED_PARAMS["seed"]) if HARDCODED_PARAMS["seed"] is not None else config.seed
    max_series = (
        int(HARDCODED_PARAMS["max_series"]) if HARDCODED_PARAMS["max_series"] is not None else config.max_series
    )
    normalize = config.normalize_per_series and not HARDCODED_PARAMS["skip_normalization"]
    show_examples = int(HARDCODED_PARAMS["show_examples"])

    print("Using forecast configuration:")
    print(f"  config path: {config.config_path}")
    print(f"  repo id: {config.repo_id}")
    print(f"  split: {config.split}")
    print(f"  target dtype: {config.target_dtype}")
    print(f"  normalize per series: {normalize}")
    print(f"  max series: {max_series}")
    print(f"  seed: {seed}")

    results: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        try:
            record = inspect_dataset(
                dataset_name,
                config=config,
                seed=seed,
                max_series=max_series,
                show_examples=show_examples,
                normalize=normalize,
            )
            results.append(record)
        except Exception as exc:  # pragma: no cover - debugging helper
            print(f"\nDataset '{dataset_name}' failed during inspection: {exc}")

    if not results:
        print("\nNo datasets were inspected.")
        return

    nan_total = sum(entry["nan_series"] for entry in results)
    inf_total = sum(entry["inf_series"] for entry in results)
    inspected_total = sum(entry["inspected_series"] for entry in results)

    print("\n=== Summary ===")
    print(f"  datasets inspected: {len(results)}")
    print(f"  total series inspected: {inspected_total}")
    print(f"  datasets with NaNs: {sum(1 for entry in results if entry['nan_series'] > 0)}")
    print(f"  datasets with Infs: {sum(1 for entry in results if entry['inf_series'] > 0)}")
    print(f"  total NaN series: {nan_total}")
    print(f"  total Inf series: {inf_total}")


if __name__ == "__main__":
    main()
