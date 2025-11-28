"""Zero-shot evaluation of Chronos-trained forecasting heads on ICML datasets."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from time_series_loader import TimeSeriesDataModule
from util import default_device
from down_tasks.forecast_shared import apply_model_overrides
from down_tasks.forecast_utils import (
    ensure_dataloader_pred_len,
    print_evaluation_summary,
    save_evaluation_results,
)
from evaluation_down_tasks.zeroshot_utils import (
    build_model_from_checkpoint,
    dataset_slug,
    determine_config_path,
    evaluate_and_collect,
    extract_checkpoint_timestamp,
    load_zeroshot_config,
    select_loader,
)

CONFIG_ENV_VAR = "ICML_ZEROSHOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "icml_zeroshot.yaml"


def _sanitize_for_json(obj: object) -> object:
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def _aggregate_results_by_horizon(
    results: Dict[str, Dict[int, Dict[str, float]]]
) -> Dict[int, Dict[str, object]]:
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


def _save_horizon_summary(
    summary: Dict[int, Dict[str, object]],
    *,
    results_dir: Path,
    prefix: str,
    timestamp: str,
) -> Tuple[Path, Path]:
    if not summary:
        raise ValueError("Horizon summary is empty; nothing to save.")

    horizon_json = results_dir / f"{prefix}_horizon_{timestamp}.json"
    horizon_csv = results_dir / f"{prefix}_horizon_{timestamp}.csv"

    with horizon_json.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_for_json(summary), handle, indent=2)

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


if __name__ == "__main__":
    config_path = determine_config_path(DEFAULT_CONFIG_PATH)
    zeroshot_cfg = load_zeroshot_config(config_path)

    base_config = tu.load_config(zeroshot_cfg.model_config_path)

    seed = zeroshot_cfg.seed if zeroshot_cfg.seed is not None else base_config.seed
    tu.set_seed(seed)
    torch.manual_seed(seed)

    horizons = zeroshot_cfg.horizons
    device = default_device()
    print(f"Using device: {device}")
    print(f"Requested horizons: {horizons}")
    print(f"Using zero-shot configuration: {zeroshot_cfg.config_path}")
    print(f"Using encoder configuration: {zeroshot_cfg.model_config_path}")

    results_dir = zeroshot_cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = apply_model_overrides(
        base_config.model,
        token_size=zeroshot_cfg.overrides.get("token_size"),
        model_dim=zeroshot_cfg.overrides.get("model_dim"),
        embedding_dim=zeroshot_cfg.overrides.get("embedding_dim"),
        depth=zeroshot_cfg.overrides.get("depth"),
    )

    model, checkpoint_info, eval_horizons, max_horizon, sequence_first_input = build_model_from_checkpoint(
        model_cfg=model_cfg,
        checkpoint_path=zeroshot_cfg.forecast_checkpoint_path,
        requested_horizons=horizons,
        device=device,
        encoder_checkpoint_path=zeroshot_cfg.encoder_checkpoint_path,
        visual_encoder_checkpoint_path=zeroshot_cfg.visual_encoder_checkpoint_path,
        force_dual=False,
    )

    checkpoint_info.update(
        {
            "encoder_config_path": str(zeroshot_cfg.model_config_path),
            "encoder_checkpoint_path": str(
                zeroshot_cfg.encoder_checkpoint_path or zeroshot_cfg.forecast_checkpoint_path
            ),
            "forecast_head_checkpoint_path": str(zeroshot_cfg.forecast_checkpoint_path),
            "zero_shot_config_path": str(zeroshot_cfg.config_path),
        }
    )

    print(
        "Loaded forecasting model with horizons "
        f"{checkpoint_info['horizons']} (evaluating {eval_horizons}), max horizon {max_horizon}"
    )

    # Generate timestamp from checkpoint name or current time
    checkpoint_timestamp = extract_checkpoint_timestamp(zeroshot_cfg.forecast_checkpoint_path)
    if checkpoint_timestamp is None:
        checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create predictions directory early
    predictions_dir = results_dir / f"{zeroshot_cfg.output_prefix}_{checkpoint_timestamp}_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    module = TimeSeriesDataModule(
        dataset_name=zeroshot_cfg.dataset_name or "",
        dataset_names=zeroshot_cfg.dataset_names,
        data_dir=zeroshot_cfg.data_dir,
        batch_size=zeroshot_cfg.batch_size,
        val_batch_size=zeroshot_cfg.val_batch_size,
        num_workers=zeroshot_cfg.num_workers,
        pin_memory=True,
        normalize=True,
        filename=zeroshot_cfg.filename,
        train=True,
        val=False,
        test=False,
    )
    dataset_groups = module.get_dataloaders()
    if not dataset_groups:
        raise RuntimeError("No datasets available for evaluation. Check filters or data directory.")

    results_by_dataset: Dict[str, Dict[int, Dict[str, float]]] = {}
    prediction_payloads: Dict[str, Dict[str, object]] = {}
    for group in dataset_groups:
        loader, _, split_used = select_loader(group, zeroshot_cfg.split)
        if loader is None:
            print(f"Skipping dataset '{group.name}' because neither val nor test splits are available.")
            continue

        try:
            ensure_dataloader_pred_len(loader, max_horizon)
            print(f"\nEvaluating dataset '{group.name}' on '{split_used}' split...")
            # Do the evaluation
            metrics, payload = evaluate_and_collect(
                model,
                loader,
                device,
                list(eval_horizons),
                max_horizon,
                sequence_first_input=sequence_first_input,
            )
            if metrics is None or payload is None:
                print(f"  No metrics computed for dataset '{group.name}'.")
                continue

            results_by_dataset[group.name] = metrics
            payload["dataset"] = group.name
            payload["dataset_slug"] = dataset_slug(group.name)
            payload["split"] = split_used
            prediction_payloads[group.name] = payload
            
            # Save predictions immediately after evaluation
            dataset_dir = predictions_dir / payload["dataset_slug"]
            dataset_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, dataset_dir / "data.pt")
            print(f"  Saved predictions to: {dataset_dir / 'data.pt'}")
            
            for horizon in eval_horizons:
                if horizon in metrics:
                    mse = metrics[horizon]["mse"]
                    mae = metrics[horizon]["mae"]
                    mse_str = f"{mse:.6f}" if not math.isnan(mse) else "nan"
                    mae_str = f"{mae:.6f}" if not math.isnan(mae) else "nan"
                    print(f"  H{horizon}: MSE={mse_str}, MAE={mae_str}")
                    
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                print(f"  Skipping dataset '{group.name}' due to out of memory error: {exc}")
                torch.cuda.empty_cache()  # Clear GPU memory if available
                continue
            else:
                raise  # Re-raise non-OOM runtime errors

    if not results_by_dataset:
        print("No datasets produced evaluation metrics. Nothing to report.")
        sys.exit(0)

    dataset_json_path, dataset_csv_path = save_evaluation_results(
        results_by_dataset,
        checkpoint_info,
        results_dir,
        prefix=zeroshot_cfg.output_prefix,
        timestamp=checkpoint_timestamp,
    )

    print_evaluation_summary(results_by_dataset, checkpoint_info)

    horizon_summary = _aggregate_results_by_horizon(results_by_dataset)
    horizon_json_path, horizon_csv_path = _save_horizon_summary(
        horizon_summary,
        results_dir=results_dir,
        prefix=zeroshot_cfg.output_prefix,
        timestamp=checkpoint_timestamp,
    )

    print("\nSaved evaluation artifacts:")
    print(f"  Per-dataset JSON: {dataset_json_path}")
    print(f"  Per-dataset CSV:  {dataset_csv_path}")
    print(f"  Per-horizon JSON: {horizon_json_path}")
    print(f"  Per-horizon CSV:  {horizon_csv_path}")
    print(f"  Predictions dir:  {predictions_dir}")

    print("\nModel artifact references:")
    print(f"  Encoder config: {zeroshot_cfg.model_config_path}")
    if zeroshot_cfg.encoder_checkpoint_path is not None:
        print(f"  Encoder checkpoint: {zeroshot_cfg.encoder_checkpoint_path}")
    if zeroshot_cfg.visual_encoder_checkpoint_path is not None:
        print(f"  Visual encoder checkpoint: {zeroshot_cfg.visual_encoder_checkpoint_path}")
    print(f"  Forecast head checkpoint: {zeroshot_cfg.forecast_checkpoint_path}")

    print("\nPer-horizon summary (mean metrics across datasets):")
    for horizon in sorted(horizon_summary.keys()):
        payload = horizon_summary[horizon]
        mse = payload["mean_mse"]
        mae = payload["mean_mae"]
        mse_str = f"{mse:.6f}" if not math.isnan(mse) else "nan"
        mae_str = f"{mae:.6f}" if not math.isnan(mae) else "nan"
        print(
            f"  H{horizon}: datasets={payload['dataset_count']}, "
            f"mean MSE={mse_str}, mean MAE={mae_str}"
        )
