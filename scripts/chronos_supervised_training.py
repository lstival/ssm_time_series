"""Train a single encoder for Chronos forecasting using YAML-configured datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import comet_ml
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ssm_time_series import training as tu
from ssm_time_series import utils as u
from ssm_time_series.data.utils import (
    build_dataloaders,
    prepare_dataset,
    resolve_checkpoint_dir,
    resolve_path,
    split_dataset,
)
from ssm_time_series.models.classifier import ChronosForecastModel


def _prepare_forecast_batch(batch, pred_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
    if seq.size(1) <= pred_len:
        raise ValueError(
            f"Patched sequence length {seq.size(1)} must be larger than prediction length {pred_len}."
        )
    context = seq[:, :-pred_len, :]
    target = seq[:, -pred_len:, :]
    return context, target


def train_one_epoch(
    model: ChronosForecastModel,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    pred_len: int,
    scaler: GradScaler,
    max_grad_norm: Optional[float],
    amp_enabled: bool,
) -> float:
    model.train()
    running = 0.0
    steps = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        context, target = _prepare_forecast_batch(batch, pred_len, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled): # Active the autocast to use mixed precision inside the training
            preds = model(context)
            loss = criterion(preds, target)
        scaler.scale(loss).backward() # Use scale fator to avoid zero (gradient vanishing) in the update pass (because float16 can round the values to 0 when too small)
        if max_grad_norm is not None:
            scaler.unscale_(optimizer) # Remove the scale factor to avoid (inf and Nan) when clipping to a correct range
            clip_grad_norm_(model.parameters(), max_grad_norm) # Create a clip in the max and min in the gradient after the unscaling process
        scaler.step(optimizer) # Verify that gradient are in normal scalle e avoid Nan values
        scaler.update() # Update the gradient 
        running += loss.item()
        steps += 1
    return running / max(1, steps)


@torch.no_grad()
def evaluate(
    model: ChronosForecastModel,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    *,
    device: torch.device,
    pred_len: int,
) -> Optional[float]:
    if loader is None:
        return None
    model.eval()
    running = 0.0
    steps = 0
    for batch in tqdm(loader, desc="Val", leave=False):
        context, target = _prepare_forecast_batch(batch, pred_len, device)
        preds = model(context)
        loss = criterion(preds, target)
        running += loss.item()
        steps += 1
    return running / max(1, steps)


def _train_encoder(
    model: ChronosForecastModel,
    *,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    training_cfg: dict,
    device: torch.device,
    epochs: int,
    pred_len: int,
    checkpoint_dir: Path,
    experiment: Optional[comet_ml.Experiment] = None,
) -> None:
    optimizer = tu.build_optimizer(model, training_cfg)
    scheduler = tu.build_scheduler(optimizer, training_cfg, epochs)
    criterion = nn.MSELoss()
    amp_enabled = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0)) or None
    scaler = GradScaler(enabled=amp_enabled)

    model.to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log hyperparameters to Comet
    if experiment is not None:
        experiment.log_parameters({
            "epochs": epochs,
            "pred_len": pred_len,
            "learning_rate": training_cfg.get("learning_rate", "N/A"),
            "weight_decay": training_cfg.get("weight_decay", 0.0),
            "batch_size": training_cfg.get("batch_size", "N/A"),
            "use_amp": amp_enabled,
            "max_grad_norm": max_grad_norm,
            "optimizer": training_cfg.get("optimizer", "AdamW"),
            "scheduler": training_cfg.get("scheduler", "None"),
            "device": str(device),
        })

    best_metric = float("inf")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            pred_len=pred_len,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            amp_enabled=amp_enabled,
        )
        val_loss = evaluate(model, val_loader, criterion, device=device, pred_len=pred_len)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss if val_loss is not None else train_loss)
        elif scheduler is not None:
            scheduler.step()

        # Log metrics to Comet
        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            experiment.log_metric("learning_rate", current_lr, step=epoch + 1)

        metric = val_loss if val_loss is not None else train_loss
        print(
            f"Epoch {epoch + 1}/{epochs} | train={train_loss:.4f}"
            + (f" | val={val_loss:.4f}" if val_loss is not None else "")
        )

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if metric < best_metric:
            best_metric = metric
            torch.save({**state, "best_metric": best_metric}, checkpoint_dir / "best.pt")


def train_temporal_encoder(
    *,
    config: tu.ExperimentConfig,
    training_cfg: dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    pred_len: int,
    feature_dim: int,
    checkpoint_dir: Path,
    experiment: Optional[comet_ml.Experiment] = None,
) -> None:
    encoder = tu.build_encoder_from_config(config.model)
    model = ChronosForecastModel(encoder, feature_dim, target_dim=feature_dim, pred_len=pred_len)
    _train_encoder(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        device=device,
        epochs=epochs,
        pred_len=pred_len,
        checkpoint_dir=checkpoint_dir,
        experiment=experiment,
    )


def train_visual_encoder(
    *,
    config: tu.ExperimentConfig,
    training_cfg: dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    pred_len: int,
    feature_dim: int,
    checkpoint_dir: Path,
    experiment: Optional[comet_ml.Experiment] = None,
) -> None:
    encoder = tu.build_visual_encoder_from_config(config.model)
    model = ChronosForecastModel(encoder, feature_dim, target_dim=feature_dim, pred_len=pred_len)
    _train_encoder(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        device=device,
        epochs=epochs,
        pred_len=pred_len,
        checkpoint_dir=checkpoint_dir,
        experiment=experiment,
    )


def main() -> None:
    # Initialize Comet ML experiment from config
    from ssm_time_series.utils.comet import create_comet_experiment
    experiment = create_comet_experiment("chronos_supervised")
    
    default_cfg = Path(__file__).resolve().parent / "configs" / "chronos_supervised.yaml"
    config_path = resolve_path(Path.cwd(), default_cfg)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {default_cfg}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")
    
    # Log configuration to Comet
    experiment.log_parameters({
        "seed": config.seed,
        "config_file": str(config_path),
    })

    data_cfg = config.data
    dataset = prepare_dataset(config_path, data_cfg)
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=config.seed)

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = val_loader if val_loader is not None and len(val_loader) > 0 else None

    sample_batch = next(iter(train_loader))
    sample_seq = u.prepare_sequence(u.extract_sequence(sample_batch))
    feature_dim = sample_seq.shape[-1]
    seq_len = sample_seq.shape[1]

    training_cfg = config.training
    pred_len = int(training_cfg.get("pred_len", 720))
    if seq_len <= pred_len:
        raise ValueError(
            f"Chronos patches ({seq_len} steps) must be longer than the requested prediction length ({pred_len})."
        )

    epochs = int(training_cfg.get("epochs", 100))

    checkpoint_root = resolve_checkpoint_dir(config, config_path, None)
    encoder_choice = str(training_cfg.get("encoder_type", "temporal")).lower()
    if encoder_choice not in {"temporal", "visual"}:
        raise ValueError("training.encoder_type must be 'temporal' or 'visual'")
    checkpoint_dir = (checkpoint_root / encoder_choice).resolve()
    print(f"Checkpoints: {checkpoint_dir}")

    # Log encoder type and dataset info to Comet
    experiment.log_parameters({
        "encoder_type": encoder_choice,
        "feature_dim": feature_dim,
        "seq_len": seq_len,
        "pred_len": pred_len,
    })
    
    trainer_kwargs = dict(
        config=config,
        training_cfg=training_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        pred_len=pred_len,
        feature_dim=feature_dim,
        checkpoint_dir=checkpoint_dir,
        experiment=experiment,
    )

    if encoder_choice == "temporal":
        train_temporal_encoder(**trainer_kwargs)
    else:
        train_visual_encoder(**trainer_kwargs)
    
    # End Comet experiment
    experiment.end()


if __name__ == "__main__":
    main()
