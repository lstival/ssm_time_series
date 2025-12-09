"""Fine-tune a dual encoder forecasting head where both encoders remain trainable."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import comet_ml
import datasets
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

import sys

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from util import (
    default_device,
    prepare_run_directory,
)
from down_tasks.forecast_shared import (
    apply_model_overrides,
    finalize_results,
)
from models.utils import (
    load_chronos_forecast_config,
)
from models.dual_forecast import DualEncoderForecastMLP, DualEncoderForecastRegressor
from dataloaders.utils import (
    ChronosDatasetGroup,
    build_dataset_group,
    _build_aggregated_loader,
)


def ensure_hf_list_feature_registered() -> None:
    """Register HuggingFace's legacy 'List' feature if missing."""
    try:
        features_module = getattr(datasets, "features", None)
        if features_module is None:
            return
        registry = getattr(features_module, "_FEATURE_TYPES", None)
        sequence_cls = getattr(features_module, "Sequence", None)
        if isinstance(registry, dict) and sequence_cls is not None and "List" not in registry:
            registry["List"] = sequence_cls
            return
        inner = getattr(features_module, "features", None)
        if inner is not None:
            registry = getattr(inner, "_FEATURE_TYPES", registry)
            sequence_cls = getattr(inner, "Sequence", sequence_cls)
            if isinstance(registry, dict) and sequence_cls is not None and "List" not in registry:
                registry["List"] = sequence_cls
    except Exception as exc:
        print(f"Warning: unable to register HuggingFace 'List' feature: {exc}")


def train_full_dual_encoder_dataset_group(
    *,
    group_name: str,
    train_loader,
    val_loader,
    encoder: nn.Module,
    visual_encoder: nn.Module,
    device: torch.device,
    horizons: List[int],
    mlp_hidden_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    run_root: Path,
    max_horizon: int,
    criterion: nn.Module,
    experiment: Optional[comet_ml.Experiment] = None,
) -> Optional[Dict[str, object]]:
    if train_loader is None:
        print(f"Skipping dataset '{group_name}' because no train loader is available.")
        return None

    try:
        sample_batch = next(iter(train_loader))
    except StopIteration:
        print(f"Skipping dataset '{group_name}' because the train loader is empty.")
        return None

    seq_x, seq_y, _, _ = sample_batch
    seq_x, seq_y = seq_x.float(), seq_y.float()
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

    encoder_embedding_dim = getattr(encoder, "embedding_dim", None)
    visual_encoder_embedding_dim = getattr(visual_encoder, "embedding_dim", None)
    if encoder_embedding_dim is None:
        raise AttributeError("Encoder is expected to expose an 'embedding_dim' attribute.")
    if visual_encoder_embedding_dim is None:
        raise AttributeError("Visual encoder is expected to expose an 'embedding_dim' attribute.")

    combined_input_dim = encoder_embedding_dim + visual_encoder_embedding_dim
    print(
        "Combined input dimension: "
        f"{combined_input_dim} (encoder: {encoder_embedding_dim} + visual: {visual_encoder_embedding_dim})"
    )

    head = DualEncoderForecastMLP(
        input_dim=combined_input_dim,
        hidden_dim=mlp_hidden_dim,
        horizons=horizons,
        target_features=target_features,
    ).to(device)

    model = DualEncoderForecastRegressor(
        encoder=encoder,
        visual_encoder=visual_encoder,
        head=head,
        freeze_encoders=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dataset_slug = group_name.replace("\\", "__").replace("/", "__") or "dataset"
    dataset_dir = run_root / dataset_slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = dataset_dir / "best_model.pt"
    print(f"Training dual encoder model on dataset '{group_name}' (artifacts -> {dataset_dir})")

    if experiment is not None:
        experiment.log_parameters(
            {
                "dataset": group_name,
                "horizons": list(horizons),
                "max_horizon": max_horizon,
                "mlp_hidden_dim": mlp_hidden_dim,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "target_features": target_features,
                "encoder_embedding_dim": encoder_embedding_dim,
                "visual_encoder_embedding_dim": visual_encoder_embedding_dim,
                "combined_input_dim": combined_input_dim,
                "trainable_parameters": trainable_params,
            }
        )

    best_metric = float("inf")
    best_state: Optional[Dict[str, object]] = None

    for epoch in range(epochs):
        print(f"\n[{group_name}] Epoch {epoch + 1}/{epochs}")

        model.train()
        train_losses: Dict[int, float] = {}
        for horizon in horizons:
            epoch_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                seq_x, seq_y, _, _ = batch
                seq_x = seq_x.float().to(device)
                seq_y = seq_y.float().to(device)

                optimizer.zero_grad()
                predictions = model(seq_x, horizon=horizon)
                targets = seq_y[:, :horizon, :]
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            train_losses[horizon] = epoch_loss / num_batches if num_batches > 0 else float("nan")

        model.eval()
        val_metrics: Dict[int, Dict[str, float]] = {}
        if val_loader is not None:
            with torch.no_grad():
                for horizon in horizons:
                    mse_sum = 0.0
                    mae_sum = 0.0
                    num_samples = 0
                    for batch in val_loader:
                        seq_x, seq_y, _, _ = batch
                        seq_x = seq_x.float().to(device)
                        seq_y = seq_y.float().to(device)

                        predictions = model(seq_x, horizon=horizon)
                        targets = seq_y[:, :horizon, :]

                        mse = torch.nn.functional.mse_loss(predictions, targets).item()
                        mae = torch.nn.functional.l1_loss(predictions, targets).item()

                        batch_size = targets.size(0)
                        mse_sum += mse * batch_size
                        mae_sum += mae * batch_size
                        num_samples += batch_size

                    if num_samples > 0:
                        val_metrics[horizon] = {
                            "mse": mse_sum / num_samples,
                            "mae": mae_sum / num_samples,
                        }
                    else:
                        val_metrics[horizon] = {"mse": float("nan"), "mae": float("nan")}

        avg_train_loss = sum(train_losses.values()) / len(train_losses)
        if val_metrics:
            valid_vals = [metrics["mse"] for metrics in val_metrics.values() if not torch.isnan(torch.tensor(metrics["mse"]))]
            avg_val_mse = sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")
        else:
            avg_val_mse = float("nan")

        train_str = ", ".join([f"H{h}: {train_losses[h]:.4f}" for h in horizons])
        print(f"  Train - {train_str}")
        if val_metrics:
            val_parts = []
            for h in horizons:
                if h in val_metrics:
                    metrics = val_metrics[h]
                    val_parts.append(f"H{h}: MSE={metrics['mse']:.4f}/MAE={metrics['mae']:.4f}")
                else:
                    val_parts.append(f"H{h}: MSE=nan/MAE=nan")
            print(f"  Val   - {', '.join(val_parts)}")
        else:
            print("  Val   - unavailable")
        print(f"  Avg Train: {avg_train_loss:.4f}, Avg Val MSE: {avg_val_mse:.4f}")

        if experiment is not None:
            for h in horizons:
                experiment.log_metric(f"{group_name}_train_loss_H{h}", train_losses[h], step=epoch + 1)
            experiment.log_metric(f"{group_name}_avg_train_loss", avg_train_loss, step=epoch + 1)
            if val_metrics:
                for h in horizons:
                    if h in val_metrics:
                        experiment.log_metric(f"{group_name}_val_mse_H{h}", val_metrics[h]["mse"], step=epoch + 1)
                        experiment.log_metric(f"{group_name}_val_mae_H{h}", val_metrics[h]["mae"], step=epoch + 1)
                experiment.log_metric(f"{group_name}_avg_val_mse", avg_val_mse, step=epoch + 1)
            experiment.log_metric(f"{group_name}_learning_rate", optimizer.param_groups[0]["lr"], step=epoch + 1)

        metric_to_compare = avg_val_mse if not torch.isnan(torch.tensor(avg_val_mse)) else avg_train_loss
        if best_state is None or metric_to_compare < best_metric:
            best_metric = metric_to_compare
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_results": val_metrics if val_metrics else None,
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
                "val_metrics": {h: dict(metrics) for h, metrics in val_metrics.items()} if val_metrics else None,
                "avg_val_mse": avg_val_mse,
                "epoch": epoch + 1,
                "target_features": int(target_features),
                "checkpoint_path": str(best_checkpoint_path),
                "best_metric": metric_to_compare,
                "trainable_parameters": trainable_params,
            }
            print(f"  → Saved best dual encoder model (metric: {best_metric:.4f})")

    return best_state


if __name__ == "__main__":
    from comet_utils import create_comet_experiment

    experiment = create_comet_experiment("full_chronos_dual_encoder")
    config = load_chronos_forecast_config()

    print(f"Training dual encoder model on horizons: {config.horizons} (max horizon: {config.max_horizon})")
    experiment.log_parameters(
        {
            "horizons": config.horizons,
            "max_horizon": config.max_horizon,
            "seed": config.seed,
        }
    )

    tu.set_seed(config.seed)
    torch.manual_seed(config.seed)

    base_config = tu.load_config(config.model_config_path)

    print(f"Using offline cache directory: {config.load_kwargs['offline_cache_dir']}")
    if config.load_kwargs.get("force_offline"):
        print("Forcing offline mode - no network access will be attempted")

    device = default_device()
    print(f"Using device: {device}")

    model_cfg = apply_model_overrides(
        base_config.model,
        token_size=config.overrides.get("token_size"),
        model_dim=config.overrides.get("model_dim"),
        embedding_dim=config.overrides.get("embedding_dim"),
        depth=config.overrides.get("depth"),
    )

    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    visual_encoder = tu.build_visual_encoder_from_config(model_cfg).to(device)

    if config.context_length < getattr(encoder, "input_dim", 1):
        raise ValueError(
            f"context_length {config.context_length} must be >= encoder token size {getattr(encoder, 'input_dim', '?')}"
        )

    print("Initializing both encoders from scratch (no checkpoint weights loaded).")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    run_root = prepare_run_directory(config.checkpoint_dir, "multi_horizon_forecast_dual_full")
    print(f"Checkpoint root directory: {run_root}")
    print(f"Results directory: {config.results_dir}")

    experiment.log_parameters(
        {
            "context_length": config.context_length,
            "stride": config.stride,
            "split": config.split,
            "batch_size": config.batch_size,
            "val_batch_size": config.val_batch_size,
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "mlp_hidden_dim": config.mlp_hidden_dim,
            "initializer": "random",
        }
    )

    torch_dtype = torch.float32
    ensure_hf_list_feature_registered()

    dataset_groups: List[ChronosDatasetGroup] = []
    for dataset_name in config.datasets_to_load:
        group = build_dataset_group(
            dataset_name,
            repo_id=config.repo_id,
            split=config.split,
            target_dtype=config.target_dtype,
            normalize_per_series=config.normalize_per_series,
            load_kwargs=config.load_kwargs,
            context_length=config.context_length,
            horizon=config.max_horizon,
            stride=config.stride,
            batch_size=config.batch_size,
            val_batch_size=config.val_batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            torch_dtype=torch_dtype,
            max_windows_per_series=config.max_windows_per_series,
            max_series=config.max_series,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
        if group is not None:
            dataset_groups.append(group)

    if not dataset_groups:
        print("No datasets were prepared. Check Chronos configuration or dataset availability.")

    dataset_records: List[Dict[str, object]] = []

    combined_train_loader = (
        _build_aggregated_loader(
            dataset_groups,
            is_train=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        if dataset_groups
        else None
    )
    combined_val_loader = (
        _build_aggregated_loader(
            dataset_groups,
            is_train=False,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        if dataset_groups
        else None
    )

    if combined_train_loader is None:
        print("No training data available after aggregating all datasets.")
    else:
        combined_metadata = {
            "datasets": [group.name for group in dataset_groups],
            "train_windows": sum(group.metadata.get("train_windows", 0) for group in dataset_groups),
            "val_windows": sum(group.metadata.get("val_windows", 0) for group in dataset_groups),
        }
        print(
            "Training dual encoder model on aggregated datasets "
            f"{combined_metadata['datasets']} -> {combined_metadata['train_windows']} windows."
        )
        if experiment is not None:
            experiment.log_parameters({"combined_train_windows": combined_metadata["train_windows"], "combined_val_windows": combined_metadata["val_windows"]})

        record = train_full_dual_encoder_dataset_group(
            group_name="all_datasets",
            train_loader=combined_train_loader,
            val_loader=combined_val_loader,
            encoder=encoder,
            visual_encoder=visual_encoder,
            device=device,
            horizons=config.horizons,
            mlp_hidden_dim=config.mlp_hidden_dim,
            lr=config.lr,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            run_root=run_root,
            max_horizon=config.max_horizon,
            criterion=criterion,
            experiment=experiment,
        )
        if record is not None:
            dataset_records.append(record)

    experiment.end()

    if not dataset_records:
        print("No datasets were trained. Check dataset filters or availability.")

    finalize_results(
        dataset_records=dataset_records,
        horizons=config.horizons,
        run_root=run_root,
        results_dir=config.results_dir,
        filename_prefix="forecast_results_dual_full",
        checkpoint_path=Path("random_init_dual_encoders"),
        encoder_embedding_dim=getattr(encoder, "embedding_dim", None),
        mlp_hidden_dim=config.mlp_hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        extra_model_config={
            "context_length": config.context_length,
            "stride": config.stride,
            "split": config.split,
            "dual_encoder": True,
            "train_encoders": True,
            "combined_embedding_dim": getattr(encoder, "embedding_dim", 0)
            + getattr(visual_encoder, "embedding_dim", 0),
        },
    )
