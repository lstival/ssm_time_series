"""Convert GiftEvalParquet (HF cache) → save_to_disk format expected by gift_eval.Dataset.

The official gift_eval lib calls datasets.load_from_disk(storage_path / name) and expects
columns: item_id (str), start (timestamp), freq (str), target (array).

Our GiftEvalParquet cache has: item_id, frequency, history_start, history_value,
future_start, future_value. We reconstruct the full series (history + future) to
match the gift_eval Dataset contract, then split via gift_eval's rolling window.

Usage:
    python gift_eval_converter.py --out_dir /path/to/gift_eval_storage
    export GIFT_EVAL=/path/to/gift_eval_storage
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import datasets
import pandas as pd


PARQUET_CACHE = "/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
REPO_ID       = "Salesforce/GiftEvalParquet"

# Subset name in Parquet → dataset name for gift_eval.Dataset
# gift_eval names datasets by their base name (no freq/term suffix)
SUBSET_MAP = {
    "m_dense_H_long":                "m_dense",
    "loop_seattle_H_long":           "loop_seattle",
    "sz_taxi_H_short":               "sz_taxi",
    "solar_H_long":                  "solar",
    "bizitobs_application_10S_long": "bizitobs_application",
    "bizitobs_l2c_H_long":           "bizitobs_l2c",
    "bizitobs_service_10S_long":     "bizitobs_service",
    "car_parts_M_short":             "car_parts",
    "jena_weather_H_long":           "jena_weather",
}


def _load_parquet_subset(subset_name: str) -> datasets.Dataset:
    os.environ["HF_DATASETS_CACHE"] = PARQUET_CACHE
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    ds = datasets.load_dataset(REPO_ID, subset_name, split="train")
    return ds


def _build_full_series(row: dict) -> dict:
    """Reconstruct full series: history + future concatenated."""
    hist = np.asarray(row["history_value"], dtype=np.float32)
    fut  = np.asarray(row["future_value"],  dtype=np.float32)
    full = np.concatenate([hist, fut])
    # NaN handling: keep NaN — gift_eval masks them with mask_invalid_label
    return full


def convert_subset(subset_name: str, out_dir: Path) -> None:
    ds_name = SUBSET_MAP[subset_name]
    out_path = out_dir / ds_name

    if out_path.exists():
        print(f"  {ds_name}: already exists, skipping")
        return

    print(f"  Converting {subset_name} → {ds_name} ...")
    raw = _load_parquet_subset(subset_name)

    # Group by base series id (strip _windowN suffix from item_id)
    # Each unique base series becomes one row in the output dataset.
    # item_id format: "0_window0/2019-10-03 00:00:00" → base = "0"
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for i in range(len(raw)):
        row = raw[i]
        iid = row["item_id"]
        base = iid.split("_window")[0]
        groups[base].append(row)

    records = []
    for base_id, rows in groups.items():
        # Sort windows by history_start to get chronological order
        rows_sorted = sorted(rows, key=lambda r: r["history_start"])
        row0 = rows_sorted[0]

        # Use the first window's history_start as series start
        freq = row0["frequency"]
        start_str = row0["history_start"]
        try:
            start = pd.Timestamp(start_str)
        except Exception:
            start = pd.Timestamp("1970-01-01")

        # Build full series from history of first window + future of last window
        # (windows overlap; simplest correct approach: take first window history +
        # all futures concatenated non-overlapping)
        hist = np.asarray(row0["history_value"], dtype=np.float32)
        # Append non-overlapping future blocks
        fut_blocks = []
        for r in rows_sorted:
            fut_blocks.append(np.asarray(r["future_value"], dtype=np.float32))
        target = np.concatenate([hist] + fut_blocks)

        records.append({
            "item_id": base_id,
            "start":   start,
            "freq":    freq,
            "target":  target.tolist(),
        })

    out_ds = datasets.Dataset.from_list(records)
    out_ds.save_to_disk(str(out_path))
    print(f"    Saved {len(out_ds)} series → {out_path}")


def main():
    p = argparse.ArgumentParser("Convert GiftEvalParquet to gift_eval from_disk format")
    p.add_argument("--out_dir", type=Path,
                   default=Path("/lustre/nobackup/WUR/AIN/stiva001/gift_eval_storage"))
    p.add_argument("--subsets", nargs="+", default=list(SUBSET_MAP.keys()))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.out_dir}")

    for subset in args.subsets:
        convert_subset(subset, args.out_dir)

    print("\nDone. Set: export GIFT_EVAL=" + str(args.out_dir))


if __name__ == "__main__":
    main()
