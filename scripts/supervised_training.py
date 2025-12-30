"""Supervised forecasting training for the lightweight Mamba encoder."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ssm_time_series.models.mamba_encoder import MambaEncoder
from tqdm import tqdm
from ssm_time_series.training.utils import (
    build_encoder_from_config,
    build_optimizer,
    build_scheduler,
    load_config,
    prepare_dataloaders,
    prepare_device,
    set_seed,
)


class ForecastModel(nn.Module):
    """Encoder backbone with a simple linear head for next-step forecasting."""

    def __init__(self, encoder: MambaEncoder, input_dim: int, target_dim: int, pred_len: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.pred_len = pred_len
        self.target_dim = target_dim
        if input_dim != encoder.input_dim:
            self.adapter: nn.Module = nn.Linear(input_dim, encoder.input_dim, bias=False)
        else:
            self.adapter = nn.Identity()
        self.head = nn.Linear(encoder.embedding_dim, pred_len * target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        embedding = self.encoder(x)
        out = self.head(embedding)
        return out.view(x.size(0), self.pred_len, self.target_dim)


def _step(
    model: ForecastModel,
    batch: tuple[torch.Tensor, ...],
    criterion: nn.Module,
    *,
    device: torch.device,
    pred_len: int,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_x = batch[0].to(device).float()
    seq_y = batch[1].to(device).float()
    target = seq_y[:, -pred_len:, :]
    model.to(device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        preds = model(seq_x)
        loss = criterion(preds, target)
    return loss, preds


def train_one_epoch(
    model: ForecastModel,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    pred_len: int,
    use_amp: bool,
    max_grad_norm: Optional[float],
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    running = 0.0
    count = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        loss, _ = _step(model, batch, criterion, device=device, pred_len=pred_len, use_amp=use_amp)
        loss_value = loss.item()
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        running += loss_value
        count += 1
        pbar.set_postfix({"loss": f"{loss_value:.4f}"})
    return running / max(1, count)


@torch.no_grad()
def evaluate(
    model: ForecastModel,
    loader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    pred_len: int,
    use_amp: bool,
) -> float:
    model.eval()
    running = 0.0
    count = 0
    for batch in loader:
        loss, _ = _step(model, batch, criterion, device=device, pred_len=pred_len, use_amp=use_amp)
        running += loss.item()
        count += 1
    return running / max(1, count)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Supervised forecasting with the Mamba encoder")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--pred-len", type=int, default=96, help="Number of future steps to predict")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional checkpoint directory override")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = args.config if args.config.is_absolute() else (Path(__file__).resolve().parent / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(config_path)
    set_seed(config.seed)
    device = prepare_device(config.device)
    print(f"Using device: {device}")

    root_dir = config_path.parent
    train_loader, val_loader = prepare_dataloaders(config, root_dir)
    sample_batch = next(iter(train_loader))
    feature_dim = sample_batch[0].shape[-1]
    target_dim = sample_batch[1].shape[-1]

    training_cfg = config.training
    pred_len = int(args.pred_len) if args.pred_len is not None else int(training_cfg.get("pred_len", 96))
    epochs = int(args.epochs) if args.epochs is not None else int(training_cfg.get("epochs", 100))

    encoder = build_encoder_from_config(config.model)
    model = ForecastModel(encoder, input_dim=feature_dim, target_dim=target_dim, pred_len=pred_len)

    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg, epochs)
    criterion = nn.MSELoss()
    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0)) or None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logging_cfg = config.logging
    checkpoint_dir = args.checkpoint_dir or Path(logging_cfg.get("checkpoint_dir", "./checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (root_dir / checkpoint_dir).resolve()
    run_id = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    checkpoint_dir = checkpoint_dir / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints: {checkpoint_dir}")

    best_val = float("inf")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            pred_len=pred_len,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            scaler=scaler,
        )
        if val_loader is not None and len(val_loader) > 0:
            val_loss = evaluate(model, val_loader, criterion, device=device, pred_len=pred_len, use_amp=use_amp)
            print(f"Epoch {epoch+1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif scheduler is not None:
                scheduler.step()
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_loss": val_loss}, checkpoint_dir / "best.pt")
        else:
            print(f"Epoch {epoch+1}/{epochs} | train={train_loss:.4f}")
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(train_loss)
            elif scheduler is not None:
                scheduler.step()
        torch.save({"epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "train_loss": train_loss}, checkpoint_dir / "last.pt")


if __name__ == "__main__":
    main()
