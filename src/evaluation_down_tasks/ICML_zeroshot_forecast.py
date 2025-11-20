"""Zero-shot evaluation of Chronos-trained forecasting heads on ICML datasets."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from moco_training import resolve_path
from time_series_loader import TimeSeriesDataModule
from util import default_device
from down_tasks.forecast_shared import apply_model_overrides, parse_horizon_values
from down_tasks.forecast_utils import (
    ensure_dataloader_pred_len,
    print_evaluation_summary,
    save_evaluation_results,
)
from models.classifier import ForecastRegressor, MultiHorizonForecastMLP

DEFAULT_CHECKPOINT = (
    r"C:\\WUR\\ssm_time_series\\checkpoints\\multi_horizon_forecast_chronos_20251119_1218\\all_datasets\\best_model.pt"
)
CONFIG_ENV_VAR = "ICML_ZEROSHOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "icml_zeroshot.yaml"


@dataclass
class ZeroShotConfig:
    config_path: Path
    model_config_path: Path
    overrides: Dict[str, object]
    forecast_checkpoint_path: Path
    encoder_checkpoint_path: Optional[Path]
    results_dir: Path
    data_dir: str
    dataset_name: str
    filename: Optional[str]
    batch_size: int
    val_batch_size: int
    num_workers: int
    horizons: List[int]
    split: str
    output_prefix: str
    seed: Optional[int]


def _normalize_horizons(raw: object) -> List[int]:
    if isinstance(raw, str):
        return parse_horizon_values(raw)
    try:
        values = [int(item) for item in raw]  # type: ignore[iterable]
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValueError("'horizons' must be a list of integers or comma-separated string") from exc
    if not values:
        raise ValueError("At least one horizon must be provided in configuration")
    return sorted(set(values))


def _resolve_required_path(base: Path, candidate: object, *, description: str) -> Path:
    if candidate is None:
        raise ValueError(f"Configuration missing required path for {description}")
    resolved = resolve_path(base, Path(candidate))
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {candidate}")
    return resolved


def _dataset_slug(name: str) -> str:
    slug = name.replace("\\", "__").replace("/", "__").strip()
    return slug or "dataset"


def _evaluate_and_collect(
    model: ForecastRegressor,
    loader,
    device: torch.device,
    horizons: Sequence[int],
    max_horizon: int,
) -> Tuple[Optional[Dict[int, Dict[str, float]]], Optional[Dict[str, object]]]:
    if loader is None:
        return None, None

    model.eval()
    running_sse = {int(h): 0.0 for h in horizons}
    running_sae = {int(h): 0.0 for h in horizons}
    running_ape = {int(h): 0.0 for h in horizons}
    counts = {int(h): 0 for h in horizons}
    total_samples = 0

    contexts: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    predictions_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            seq_x, seq_y, _, _ = batch
            seq_x_cpu = seq_x.float()
            seq_x_enc = seq_x_cpu.to(device).transpose(1, 2)
            seq_y_cpu = seq_y.float()

            if seq_y_cpu.size(1) < max_horizon:
                raise ValueError(
                    f"Target sequence length {seq_y_cpu.size(1)} smaller than required max horizon {max_horizon}"
                )

            target_slice = seq_y_cpu[:, -max_horizon:, :].to(device)
            preds = model(seq_x_enc)
            batch_size = target_slice.size(0)
            total_samples += batch_size

            epsilon = 1e-8
            for horizon in horizons:
                horizon = int(horizon)
                pred_sub = preds[:, :horizon, :]
                target_sub = target_slice[:, :horizon, :]
                diff = pred_sub - target_sub

                sse = torch.sum(diff.pow(2)).item()
                sae = torch.sum(torch.abs(diff)).item()
                ape = torch.sum(torch.abs(diff) / (torch.abs(target_sub) + epsilon)).item()
                elements = diff.numel()

                running_sse[horizon] += sse
                running_sae[horizon] += sae
                running_ape[horizon] += ape
                counts[horizon] += elements

            contexts.append(seq_x_cpu)
            targets_list.append(target_slice.cpu())
            predictions_list.append(preds.cpu())

    if not contexts:
        return None, None

    results: Dict[int, Dict[str, float]] = {}
    for horizon in horizons:
        horizon = int(horizon)
        if counts[horizon] == 0:
            results[horizon] = {
                "mse": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "samples": total_samples,
            }
        else:
            mse = running_sse[horizon] / counts[horizon]
            mae = running_sae[horizon] / counts[horizon]
            rmse = math.sqrt(mse)
            mape = (running_ape[horizon] / counts[horizon]) * 100.0
            results[horizon] = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "samples": total_samples,
            }

    context_tensor = torch.cat(contexts, dim=0).contiguous()
    targets_tensor = torch.cat(targets_list, dim=0).contiguous()
    predictions_tensor = torch.cat(predictions_list, dim=0).contiguous()

    per_horizon: Dict[int, Dict[str, torch.Tensor]] = {}
    for horizon in horizons:
        horizon = int(horizon)
        per_horizon[horizon] = {
            "targets": targets_tensor[:, :horizon, :].clone(),
            "predictions": predictions_tensor[:, :horizon, :].clone(),
        }

    payload = {
        "context": context_tensor.cpu(),
        "targets": targets_tensor.cpu(),
        "predictions": predictions_tensor.cpu(),
        "eval_horizons": [int(h) for h in horizons],
        "max_horizon": int(max_horizon),
        "context_length": int(context_tensor.size(1)),
        "target_features": int(context_tensor.size(2)),
        "per_horizon": {h: {key: tensor.cpu() for key, tensor in values.items()} for h, values in per_horizon.items()},
    }

    return results, payload


def _resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    if candidate is None:
        return None
    resolved = resolve_path(base, Path(candidate))
    return resolved


def _determine_config_path() -> Path:
    candidate = DEFAULT_CONFIG_PATH

    resolved = resolve_path(Path.cwd(), candidate)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Zero-shot configuration file not found: {candidate}")
    return resolved


def load_zeroshot_config(config_path: Path) -> ZeroShotConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    config_dir = config_path.parent

    model_section = dict(payload.get("model") or {})
    model_config_path = _resolve_required_path(
        config_dir,
        model_section.get("config"),
        description="model configuration",
    )
    overrides = dict(model_section.get("overrides") or {})

    paths_section = dict(payload.get("paths") or {})
    forecast_checkpoint_candidate = paths_section.get("forecast_checkpoint", DEFAULT_CHECKPOINT)
    forecast_checkpoint_path = _resolve_required_path(
        config_dir,
        forecast_checkpoint_candidate,
        description="forecast head checkpoint",
    )
    encoder_checkpoint_path = _resolve_optional_path(config_dir, paths_section.get("encoder_checkpoint"))

    results_candidate = paths_section.get("results_dir")
    if results_candidate is None:
        results_dir = (ROOT_DIR / "results").resolve()
    else:
        results_dir = resolve_path(config_dir, Path(results_candidate))
        if results_dir is None:
            results_dir = Path(results_candidate).expanduser().resolve()

    data_section = dict(payload.get("data") or {})
    data_dir_value = data_section.get("data_dir")
    if data_dir_value is None:
        data_dir = str((ROOT_DIR / "ICML_datasets").resolve())
    else:
        resolved_data_dir = resolve_path(config_dir, Path(data_dir_value))
        if resolved_data_dir is None:
            resolved_data_dir = Path(data_dir_value).expanduser().resolve()
        data_dir = str(resolved_data_dir)

    dataset_name = str(data_section.get("dataset_name", "") or "")
    filename_value = data_section.get("filename")
    if filename_value:
        resolved_filename = resolve_path(config_dir, Path(filename_value))
        filename = str(resolved_filename) if resolved_filename is not None else str(Path(filename_value).expanduser().resolve())
    else:
        filename = None

    batch_size = int(data_section.get("batch_size", 32))
    val_batch_size = int(data_section.get("val_batch_size", batch_size))
    num_workers = int(data_section.get("num_workers", 4))

    evaluation_section = dict(payload.get("evaluation") or {})
    horizons = _normalize_horizons(evaluation_section.get("horizons", [96, 192, 336, 720]))
    split = str(evaluation_section.get("split", "test")).lower()
    if split not in {"test", "val"}:
        raise ValueError("'evaluation.split' must be either 'test' or 'val'")
    output_prefix = str(evaluation_section.get("output_prefix", "icml_zeroshot_forecast"))

    seed_value = payload.get("seed")
    if seed_value is None:
        seed_value = evaluation_section.get("seed")
    seed = int(seed_value) if seed_value is not None else None

    results_dir = results_dir.expanduser().resolve()

    return ZeroShotConfig(
        config_path=config_path,
        model_config_path=model_config_path,
        overrides=overrides,
        forecast_checkpoint_path=forecast_checkpoint_path,
        encoder_checkpoint_path=encoder_checkpoint_path,
        results_dir=results_dir,
        data_dir=data_dir,
        dataset_name=dataset_name,
        filename=filename,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        horizons=horizons,
        split=split,
        output_prefix=output_prefix,
        seed=seed,
    )


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


def _build_model_from_checkpoint(
    *,
    model_cfg: Dict[str, object],
    checkpoint_path: Path,
    requested_horizons: Sequence[int],
    device: torch.device,
) -> Tuple[ForecastRegressor, Dict[str, object], List[int], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    head_weight_key = "head.shared_layers.0.weight"
    if head_weight_key not in state_dict:
        raise KeyError(
            "Checkpoint missing head weights; expected unified multi-horizon head entries."
        )

    hidden_dim = state_dict[head_weight_key].shape[0]
    input_dim = state_dict[head_weight_key].shape[1]
    if hasattr(encoder, "embedding_dim") and int(encoder.embedding_dim) != int(input_dim):
        raise ValueError(
            "Encoder embedding dimension does not match head input dimension: "
            f"{encoder.embedding_dim} != {input_dim}"
        )

    ckpt_horizons = checkpoint.get("horizons")
    if ckpt_horizons is None:
        ckpt_horizons = list(requested_horizons)
    ckpt_horizons = sorted({int(h) for h in ckpt_horizons})

    requested_set = [int(h) for h in requested_horizons]
    evaluation_horizons = [h for h in requested_set if h in ckpt_horizons]
    if not evaluation_horizons:
        evaluation_horizons = list(ckpt_horizons)
        print(
            "Warning: requested horizons not found in checkpoint; falling back to checkpoint horizons: "
            f"{evaluation_horizons}"
        )

    target_features = int(checkpoint.get("target_features", 1))
    max_horizon = int(checkpoint.get("max_horizon", max(ckpt_horizons)))

    head = MultiHorizonForecastMLP(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        horizons=list(ckpt_horizons),
        target_features=target_features,
    ).to(device)

    model = ForecastRegressor(encoder=encoder, head=head, freeze_encoder=True).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing weights when loading checkpoint: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected weights when loading checkpoint: {sorted(unexpected)}")

    checkpoint_info = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "avg_val_mse": checkpoint.get("avg_val_mse"),
        "train_losses": checkpoint.get("train_losses"),
        "val_results": checkpoint.get("val_results"),
        "horizons": list(ckpt_horizons),
        "target_features": target_features,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "max_horizon": max_horizon,
        "encoder_embedding_dim": getattr(encoder, "embedding_dim", None),
    }
    return model, checkpoint_info, list(evaluation_horizons), max_horizon


def _select_loader(group, preferred_split: str):
    if preferred_split == "test":
        primary, fallback = group.test, group.val
        primary_name, fallback_name = "test", "val"
    else:
        primary, fallback = group.val, group.test
        primary_name, fallback_name = "val", "test"

    if primary is not None:
        return primary, primary_name
    if fallback is not None:
        print(
            f"  Preferred split '{primary_name}' unavailable for {group.name}; using '{fallback_name}' split instead."
        )
        return fallback, fallback_name
    return None, primary_name


if __name__ == "__main__":
    config_path = _determine_config_path()
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

    model, checkpoint_info, eval_horizons, max_horizon = _build_model_from_checkpoint(
        model_cfg=model_cfg,
        checkpoint_path=zeroshot_cfg.forecast_checkpoint_path,
        requested_horizons=horizons,
        device=device,
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

    module = TimeSeriesDataModule(
        dataset_name=zeroshot_cfg.dataset_name or "",
        data_dir=zeroshot_cfg.data_dir,
        batch_size=zeroshot_cfg.batch_size,
        val_batch_size=zeroshot_cfg.val_batch_size,
        num_workers=zeroshot_cfg.num_workers,
        pin_memory=True,
        normalize=True,
        filename=zeroshot_cfg.filename,
        train=False,
        val=True,
        test=True,
    )
    dataset_groups = module.get_dataloaders()
    if not dataset_groups:
        raise RuntimeError("No datasets available for evaluation. Check filters or data directory.")

    results_by_dataset: Dict[str, Dict[int, Dict[str, float]]] = {}
    prediction_payloads: Dict[str, Dict[str, object]] = {}
    for group in dataset_groups:
        loader, split_used = _select_loader(group, zeroshot_cfg.split)
        if loader is None:
            print(f"Skipping dataset '{group.name}' because neither val nor test splits are available.")
            continue

        ensure_dataloader_pred_len(loader, max_horizon)
        print(f"\nEvaluating dataset '{group.name}' on '{split_used}' split...")
        metrics, payload = _evaluate_and_collect(model, loader, device, list(eval_horizons), max_horizon)
        if metrics is None or payload is None:
            print(f"  No metrics computed for dataset '{group.name}'.")
            continue

        results_by_dataset[group.name] = metrics
        payload["dataset"] = group.name
        payload["dataset_slug"] = _dataset_slug(group.name)
        prediction_payloads[group.name] = payload
        for horizon in eval_horizons:
            if horizon in metrics:
                mse = metrics[horizon]["mse"]
                mae = metrics[horizon]["mae"]
                mse_str = f"{mse:.6f}" if not math.isnan(mse) else "nan"
                mae_str = f"{mae:.6f}" if not math.isnan(mae) else "nan"
                print(f"  H{horizon}: MSE={mse_str}, MAE={mae_str}")

    if not results_by_dataset:
        print("No datasets produced evaluation metrics. Nothing to report.")
        sys.exit(0)

    dataset_json_path, dataset_csv_path = save_evaluation_results(
        results_by_dataset,
        checkpoint_info,
        results_dir,
        prefix=zeroshot_cfg.output_prefix,
    )

    print_evaluation_summary(results_by_dataset, checkpoint_info)

    timestamp = dataset_json_path.stem
    prefix_tag = f"{zeroshot_cfg.output_prefix}_"
    if timestamp.startswith(prefix_tag):
        timestamp = timestamp[len(prefix_tag) :]
    horizon_summary = _aggregate_results_by_horizon(results_by_dataset)
    horizon_json_path, horizon_csv_path = _save_horizon_summary(
        horizon_summary,
        results_dir=results_dir,
        prefix=zeroshot_cfg.output_prefix,
        timestamp=timestamp,
    )

    predictions_dir = results_dir / f"{zeroshot_cfg.output_prefix}_{timestamp}_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, payload in prediction_payloads.items():
        dataset_dir = predictions_dir / payload["dataset_slug"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, dataset_dir / "data.pt")

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
