"""Contrastive self-supervised training for the lightweight Mamba encoder."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.mamba_encoder import MambaEncoder
from training_utils import (
    build_encoder_from_config,
    build_optimizer,
    build_scheduler,
    infer_feature_dim,
    load_config,
    prepare_dataloaders,
    prepare_device,
    set_seed,
)


class ContrastiveModel(nn.Module):
    """Wrap the encoder with an adapter so any feature dimension is accepted."""

    def __init__(self, encoder: MambaEncoder, input_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        if input_dim != encoder.input_dim:
            self.adapter: nn.Module = nn.Linear(input_dim, encoder.input_dim, bias=False)
        else:
            self.adapter = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.encoder(x)


def _random_time_mask(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    if drop_prob <= 0.0:
        return x
    mask = torch.rand(x.shape[:2], device=x.device).unsqueeze(-1)
    keep = (mask > drop_prob).float()
    return x * keep


def _random_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def _random_scaling(x: torch.Tensor, low: float = 0.9, high: float = 1.1) -> torch.Tensor:
    scales = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(low, high)
    return x * scales


def create_contrastive_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    view1 = _random_scaling(_random_time_mask(_random_jitter(x.clone())))
    view2 = _random_scaling(_random_time_mask(_random_jitter(x.clone())))
    return view1, view2


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss1 + loss2)


class ContrastiveTrainer:
    def __init__(
        self,
        model: ContrastiveModel,
        optimizer: Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        *,
        device: torch.device,
        temperature: float = 0.2,
        use_amp: bool = False,
        max_grad_norm: Optional[float] = None,
        checkpoint_dir: Optional[Path] = None,
        save_best_only: bool = True,
        save_last: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.temperature = temperature
        self.use_amp = use_amp and device.type == "cuda"
        self.max_grad_norm = max_grad_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.best_val = float("inf")
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _forward_batch(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        seq_x = batch[0].to(self.device).float()
        if seq_x.ndim == 2:
            seq_x = seq_x.unsqueeze(0)
        view1, view2 = create_contrastive_views(seq_x)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            z1 = self.model(view1)
            z2 = self.model(view2)
            loss = info_nce_loss(z1, z2, self.temperature)
        return loss

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int) -> None:
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, train=False)
                print(f"Epoch {epoch+1}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
                self._step_scheduler(val_loss)
                self._save(epoch, val_loss)
            else:
                print(f"Epoch {epoch+1}/{epochs} | train={train_loss:.4f}")
                self._step_scheduler(train_loss)
                self._save(epoch, train_loss)

    def _run_epoch(self, loader: DataLoader, *, train: bool) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()
        running = 0.0
        count = 0
        for batch in loader:
            loss = self._forward_batch(batch)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            running += loss.item()
            count += 1
        return running / max(1, count)

    def _step_scheduler(self, metric: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _save(self, epoch: int, metric: float) -> None:
        if self.checkpoint_dir is None:
            return
        state = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metric": metric,
        }
        if self.save_best_only:
            if metric < self.best_val:
                self.best_val = metric
                torch.save(state, self.checkpoint_dir / "best.pt")
        else:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch+1}.pt")
        if self.save_last:
            torch.save(state, self.checkpoint_dir / "last.pt")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Contrastive training for the Mamba encoder")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--temperature", type=float, default=None, help="InfoNCE temperature override")
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
    feature_dim = infer_feature_dim(train_loader)

    encoder = build_encoder_from_config(config.model)
    model = ContrastiveModel(encoder, input_dim=feature_dim)

    training_cfg = config.training
    epochs = int(args.epochs) if args.epochs is not None else int(training_cfg.get("epochs", 100))
    temperature = float(args.temperature) if args.temperature is not None else float(training_cfg.get("temperature", 0.2))
    optimizer = build_optimizer(model, training_cfg)
    scheduler = build_scheduler(optimizer, training_cfg, epochs)

    logging_cfg = config.logging
    checkpoint_dir = args.checkpoint_dir or Path(logging_cfg.get("checkpoint_dir", "./checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (root_dir / checkpoint_dir).resolve()
    run_id = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    checkpoint_dir = checkpoint_dir / run_id
    print(f"Checkpoints: {checkpoint_dir}")

    trainer = ContrastiveTrainer(
        model,
        optimizer,
        scheduler,
        device=device,
    temperature=temperature,
        use_amp=bool(training_cfg.get("use_amp", False)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 0.0)) or None,
        checkpoint_dir=checkpoint_dir,
        save_best_only=bool(logging_cfg.get("save_best_only", True)),
        save_last=bool(logging_cfg.get("save_last", True)),
    )

    val_loader = val_loader if val_loader is not None and len(val_loader) > 0 else None
    trainer.fit(train_loader, val_loader, epochs=epochs)


if __name__ == "__main__":
    main()
