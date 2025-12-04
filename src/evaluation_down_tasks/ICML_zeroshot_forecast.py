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
from evaluation_down_tasks.zeroshot_reporting import (
    aggregate_results_by_horizon,
    save_horizon_summary,
)

CONFIG_ENV_VAR = "ICML_ZEROSHOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "icml_zeroshot.yaml"



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
        visual_encoder_checkpoint_path=None,
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
        loader, dataset_obj, split_used = select_loader(group, zeroshot_cfg.split)
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
                dataset=dataset_obj,
                dataset_name=group.name,
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

    horizon_summary = aggregate_results_by_horizon(results_by_dataset)
    horizon_json_path, horizon_csv_path = save_horizon_summary(
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
