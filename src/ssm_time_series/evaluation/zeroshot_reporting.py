"""Reporting helpers shared by zero-shot evaluation scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def sanitize_for_json(obj: object) -> object:
    """Convert NaNs inside nested structures so json.dump succeeds."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    return obj


def aggregate_results_by_horizon(
    results: Dict[str, Dict[int, Dict[str, float]]]
) -> Dict[int, Dict[str, object]]:
    """Group per-dataset metrics by horizon and compute summary statistics."""
    grouped: Dict[int, List[Tuple[str, Dict[str, float]]]] = {}
    for dataset_name, horizon_metrics in results.items():
        for horizon, metrics in horizon_metrics.items():
            grouped.setdefault(int(horizon), []).append((dataset_name, metrics))

    summary: Dict[int, Dict[str, object]] = {}
    for horizon, entries in grouped.items():
        valid = [
            (dataset, metrics)
            for dataset, metrics in entries
            if not math.isnan(metrics.get("mse", float("nan")))
            and not math.isnan(metrics.get("mae", float("nan")))
        ]
        dataset_count = len(valid)
        if dataset_count == 0:
            summary[horizon] = {
                "dataset_count": 0,
                "mean_mse": float("nan"),
                "mean_mae": float("nan"),
                "min_mse": float("nan"),
                "max_mse": float("nan"),
                "min_mae": float("nan"),
                "max_mae": float("nan"),
                "datasets": [
                    {
                        "dataset": dataset,
                        "mse": metrics.get("mse", float("nan")),
                        "mae": metrics.get("mae", float("nan")),
                    }
                    for dataset, metrics in entries
                ],
            }
            continue

        mse_values = [metrics["mse"] for _, metrics in valid]
        mae_values = [metrics["mae"] for _, metrics in valid]
        summary[horizon] = {
            "dataset_count": dataset_count,
            "mean_mse": sum(mse_values) / dataset_count,
            "mean_mae": sum(mae_values) / dataset_count,
            "min_mse": min(mse_values),
            "max_mse": max(mse_values),
            "min_mae": min(mae_values),
            "max_mae": max(mae_values),
            "datasets": [
                {
                    "dataset": dataset,
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                }
                for dataset, metrics in entries
            ],
        }
    return summary


def save_horizon_summary(
    summary: Dict[int, Dict[str, object]],
    *,
    results_dir: Path,
    prefix: str,
    timestamp: str,
) -> Tuple[Path, Path]:
    """Persist aggregated horizon metrics to JSON and CSV files."""
    if not summary:
        raise ValueError("Horizon summary is empty; nothing to save.")

    horizon_json = results_dir / f"{prefix}_horizon_{timestamp}.json"
    horizon_csv = results_dir / f"{prefix}_horizon_{timestamp}.csv"

    with horizon_json.open("w", encoding="utf-8") as handle:
        import json

        json.dump(sanitize_for_json(summary), handle, indent=2)

    rows = []
    for horizon in sorted(summary.keys()):
        payload = summary[horizon]
        rows.append(
            {
                "horizon": horizon,
                "dataset_count": payload["dataset_count"],
                "mean_mse": payload["mean_mse"],
                "mean_mae": payload["mean_mae"],
                "min_mse": payload["min_mse"],
                "max_mse": payload["max_mse"],
                "min_mae": payload["min_mae"],
                "max_mae": payload["max_mae"],
            }
        )
    pd.DataFrame(rows).to_csv(horizon_csv, index=False)
    return horizon_json, horizon_csv


__all__ = [
    "aggregate_results_by_horizon",
    "save_horizon_summary",
    "sanitize_for_json",
]
