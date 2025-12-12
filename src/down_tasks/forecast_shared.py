"""Shared helpers for training forecasting heads across different datasets."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import comet_ml
import torch
import torch.nn as nn
from torch.optim import AdamW

from down_tasks.forecast_utils import ensure_dataloader_pred_len
from models.classifier import ForecastRegressor, MultiHorizonForecastMLP
from util import evaluate_dataset, train_epoch_dataset


def parse_horizon_values(raw: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Failed to parse horizons '{raw}': {exc}") from exc
    if not values:
        raise ValueError("At least one forecast horizon must be provided")
    if any(value <= 0 for value in values):
        raise ValueError(f"All horizons must be positive integers, received: {values}")
    return sorted(set(values))


def apply_model_overrides(
    base_cfg: Dict[str, object],
    *,
    token_size: Optional[int] = None,
    model_dim: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    depth: Optional[int] = None,
    default_pooling: str = "mean",
    default_dropout: float = 0.1,
) -> Dict[str, object]:
    cfg = dict(base_cfg)
    if token_size is not None:
        cfg["input_dim"] = token_size
    if model_dim is not None:
        cfg["model_dim"] = model_dim
    if embedding_dim is not None:
        cfg["embedding_dim"] = embedding_dim
    if depth is not None:
        cfg["depth"] = depth
    cfg.setdefault("pooling", default_pooling)
    cfg.setdefault("dropout", default_dropout)
    return cfg


def train_dataset_group(
    *,
    group_name: str,
    train_loader,
    val_loader,
    encoder: nn.Module,
    device: torch.device,
    horizons: Sequence[int],
    mlp_hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    run_root: Path,
    max_horizon: int,
    criterion: nn.Module,
    experiment: Optional[comet_ml.Experiment] = None,
    reverse_normalization: bool = False,
) -> Optional[Dict[str, object]]:
    if train_loader is None:
        print(f"Skipping dataset '{group_name}' because no train loader is available.")
        return None

    ensure_dataloader_pred_len(train_loader, max_horizon)
    if val_loader is not None:
        ensure_dataloader_pred_len(val_loader, max_horizon)

    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        print(f"Skipping dataset '{group_name}' because the train loader is empty.")
        return None

    seq_x, seq_y = sample_batch[0], sample_batch[1]
    target_features = seq_y.size(2)
    available_steps = seq_y.size(1)
    if available_steps < max_horizon:
        raise ValueError(
            f"Dataset '{group_name}' provides only {available_steps} forecast steps, "
            f"but max horizon {max_horizon} was requested."
        )
    print(
        f"Dataset '{group_name}': seq_x shape {tuple(seq_x.shape)}, seq_y shape {tuple(seq_y.shape)}, "
        f"target features {target_features}"
    )

    embedding_dim = getattr(encoder, "embedding_dim", None)
    if embedding_dim is None:
        raise AttributeError("Encoder is expected to expose an 'embedding_dim' attribute.")

    head = MultiHorizonForecastMLP(
        input_dim=int(embedding_dim),
        hidden_dim=mlp_hidden_dim,
        horizons=horizons,
        target_features=target_features,
    ).to(device)
    model = ForecastRegressor(encoder=encoder, head=head, freeze_encoder=True).to(device)
    for param in model.encoder.parameters():
        param.requires_grad = True
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)

    dataset_slug = group_name.replace("\\", "__").replace("/", "__") or "dataset"
    dataset_dir = run_root / dataset_slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = dataset_dir / "best_model.pt"
    print(f"Training dataset '{group_name}' (artifacts -> {dataset_dir})")

    best_metric = float("inf")
    best_state: Optional[Dict[str, object]] = None
    last_train_losses: Dict[int, float] = {h: float("nan") for h in horizons}
    last_val_metrics: Optional[Dict[int, Dict[str, float]]] = None
    last_avg_train = float("nan")
    last_avg_val = float("nan")
    
    # Log hyperparameters to Comet
    if experiment is not None:
        experiment.log_parameters({
            "dataset": group_name,
            "horizons": list(horizons),
            "max_horizon": max_horizon,
            "mlp_hidden_dim": mlp_hidden_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "target_features": target_features,
            "embedding_dim": embedding_dim,
        })

    for epoch in range(epochs):
        print(f"\n[{group_name}] Epoch {epoch + 1}/{epochs}")
        train_losses = train_epoch_dataset(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            list(horizons),
            reverse_normalize=reverse_normalization,
        )
        val_metrics = evaluate_dataset(
            model,
            val_loader,
            device,
            list(horizons),
            reverse_normalize=reverse_normalization,
        )

        avg_train_loss = sum(train_losses.values()) / len(train_losses)
        if val_metrics is not None:
            valid_vals = [metrics["mse"] for metrics in val_metrics.values() if not math.isnan(metrics["mse"])]
            avg_val_mse = sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")
        else:
            avg_val_mse = float("nan")

        train_str = ", ".join([f"H{h}: {train_losses[h]:.4f}" for h in horizons])
        print(f"  Train - {train_str}")
        if val_metrics is not None:
            val_parts = []
            for h in horizons:
                if h in val_metrics:
                    metrics = val_metrics[h]
                    val_parts.append(f"H{h}: MSE={metrics['mse']:.4f}/MAE={metrics['mae']:.4f}")
                else:
                    val_parts.append(f"H{h}: MSE=nan/MAE=nan")
            val_str = ", ".join(val_parts)
            print(f"  Val   - {val_str}")
        else:
            print("  Val   - unavailable")
        print(f"  Avg Train: {avg_train_loss:.4f}, Avg Val MSE: {avg_val_mse:.4f}")
        
        # Log metrics to Comet
        if experiment is not None:
            # Log per-horizon training losses
            for h in horizons:
                experiment.log_metric(f"{group_name}_train_loss_H{h}", train_losses[h], step=epoch + 1)
            # Log average training loss
            experiment.log_metric(f"{group_name}_avg_train_loss", avg_train_loss, step=epoch + 1)
            
            # Log validation metrics if available
            if val_metrics is not None:
                for h in horizons:
                    if h in val_metrics:
                        experiment.log_metric(f"{group_name}_val_mse_H{h}", val_metrics[h]["mse"], step=epoch + 1)
                        experiment.log_metric(f"{group_name}_val_mae_H{h}", val_metrics[h]["mae"], step=epoch + 1)
                # Log average validation MSE
                experiment.log_metric(f"{group_name}_avg_val_mse", avg_val_mse, step=epoch + 1)
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            experiment.log_metric(f"{group_name}_learning_rate", current_lr, step=epoch + 1)

        metric_to_compare = avg_val_mse
        if math.isnan(metric_to_compare):
            metric_to_compare = avg_train_loss

        if best_state is None or metric_to_compare < best_metric:
            best_metric = metric_to_compare
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_results": val_metrics,
                "avg_train_loss": avg_train_loss,
                "avg_val_mse": avg_val_mse,
                "horizons": list(horizons),
                "max_horizon": max_horizon,
                "target_features": target_features,
                "dataset": group_name,
            }
            torch.save(checkpoint, best_checkpoint_path)
            best_state = {
                "dataset": group_name,
                "dataset_slug": dataset_slug,
                "dataset_dir": str(dataset_dir),
                "train_losses": dict(train_losses),
                "avg_train_loss": avg_train_loss,
                "val_metrics": {h: dict(metrics) for h, metrics in (val_metrics or {}).items()} if val_metrics is not None else None,
                "avg_val_mse": avg_val_mse,
                "epoch": epoch + 1,
                "target_features": int(target_features),
                "checkpoint_path": str(best_checkpoint_path),
                "best_metric": metric_to_compare,
            }
            print(f"  → Saved best model (metric: {best_metric:.4f})")

        last_train_losses = dict(train_losses)
        last_val_metrics = val_metrics
        last_avg_train = avg_train_loss
        last_avg_val = avg_val_mse

    if best_state is None:
        best_state = {
            "dataset": group_name,
            "dataset_slug": dataset_slug,
            "dataset_dir": str(dataset_dir),
            "train_losses": dict(last_train_losses),
            "avg_train_loss": last_avg_train,
            "val_metrics": {h: dict(metrics) for h, metrics in (last_val_metrics or {}).items()} if last_val_metrics is not None else None,
            "avg_val_mse": last_avg_val,
            "epoch": epochs,
            "target_features": int(target_features),
            "checkpoint_path": str(best_checkpoint_path),
            "best_metric": best_metric,
        }
        if not best_checkpoint_path.exists():
            checkpoint = {
                "epoch": epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": last_train_losses,
                "val_results": last_val_metrics,
                "avg_train_loss": last_avg_train,
                "avg_val_mse": last_avg_val,
                "horizons": list(horizons),
                "max_horizon": max_horizon,
                "target_features": target_features,
                "dataset": group_name,
            }
            torch.save(checkpoint, best_checkpoint_path)

    best_metric_display = best_state["avg_val_mse"]
    if math.isnan(best_metric_display):
        best_metric_display = best_state["avg_train_loss"]
    print(f"\nFinished dataset '{group_name}'. Best metric: {best_metric_display:.4f}")
    print(f"Artifacts saved to: {dataset_dir}")
    return best_state


def finalize_results(
    *,
    dataset_records: Sequence[Dict[str, object]],
    horizons: Sequence[int],
    run_root: Path,
    results_dir: Path,
    filename_prefix: str,
    checkpoint_path: Path,
    encoder_embedding_dim: Optional[int],
    mlp_hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    val_batch_size: int,
    extra_model_config: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path]:
    print("\nTraining summary:")
    for record in dataset_records:
        metric = record.get("avg_val_mse", float("nan"))
        if math.isnan(metric):
            metric = record.get("avg_train_loss", float("nan"))
        print(f"  {record['dataset']}: best epoch {record['epoch']} (metric {metric:.4f})")

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<30} {'Horizon':<8} {'TrainLoss':<12} {'ValMSE':<12} {'ValMAE':<12}")
    print("-" * 80)
    for record in dataset_records:
        train_losses = record["train_losses"]
        val_metrics = record.get("val_metrics")
        for horizon in horizons:
            train_loss = train_losses.get(horizon, float("nan"))
            if val_metrics is not None and horizon in val_metrics:
                val_mse = val_metrics[horizon]["mse"]
                val_mae = val_metrics[horizon]["mae"]
            else:
                val_mse = float("nan")
                val_mae = float("nan")
            print(f"{record['dataset']:<30} {horizon:<8} {train_loss:<12.6f} {val_mse:<12.6f} {val_mae:<12.6f}")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"{filename_prefix}_{timestamp}.json"
    csv_path = results_dir / f"{filename_prefix}_{timestamp}.csv"

    model_config = {
        "encoder_checkpoint": str(checkpoint_path),
        "embedding_dim": encoder_embedding_dim,
        "hidden_dim": mlp_hidden_dim,
        "horizons": list(horizons),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
    }
    if extra_model_config:
        model_config.update(extra_model_config)

    results_payload = {
        "timestamp": timestamp,
        "checkpoint_root": str(run_root),
        "model_config": model_config,
        "datasets": list(dataset_records),
    }

    def _sanitize_for_json(obj: object) -> object:
        if isinstance(obj, float):
            return None if math.isnan(obj) else obj
        if isinstance(obj, dict):
            return {key: _sanitize_for_json(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(item) for item in obj]
        return obj

    with open(json_path, "w", encoding="utf-8") as fp:
        import json

        json.dump(_sanitize_for_json(results_payload), fp, indent=2)

    import pandas as pd


    csv_rows = []
    for record in dataset_records:
        val_metrics = record.get("val_metrics")
        for horizon in horizons:
            row = {
                "dataset": record["dataset"],
                "dataset_slug": record["dataset_slug"],
                "horizon": horizon,
                "train_loss": record["train_losses"].get(horizon, float("nan")),
                "val_mse": float("nan"),
                "val_mae": float("nan"),
                "avg_train_loss": record["avg_train_loss"],
                "avg_val_mse": record.get("avg_val_mse", float("nan")),
                "best_epoch": record["epoch"],
                "target_features": record["target_features"],
                "checkpoint_path": record["checkpoint_path"],
            }
            if val_metrics is not None and horizon in val_metrics:
                row["val_mse"] = val_metrics[horizon]["mse"]
                row["val_mae"] = val_metrics[horizon]["mae"]
            csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print("\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    return json_path, csv_path
