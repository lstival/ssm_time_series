"""BYOL-style single-encoder self-supervised training.

Bootstrap Your Own Latent (BYOL) trains a single encoder without negative pairs.
An online network (encoder + projector + predictor) is trained to predict the
output of a momentum (target) network given two augmented views of the same input.

Supports both temporal and visual (RP) encoders — same modes as simclr_training.py
so that probe_lotsa_checkpoint.py can load checkpoints identically.

  --mode temporal   →  MambaEncoder trained with BYOL (two noisy temporal views)
  --mode visual     →  UpperTriDiagSimCLREncoder trained with BYOL (two noisy RP views)

Key differences vs SimCLR:
  - No negative pairs needed → stable with small batches
  - Momentum (EMA) target network prevents collapse without negatives
  - Predictor head on online side only
  - Loss: negative cosine similarity (symmetric)

Usage
-----
    python3 src/byol_training.py \
        --config src/configs/lotsa_byol_temporal.yaml \
        --mode temporal

    python3 src/byol_training.py \
        --config src/configs/lotsa_byol_visual.yaml \
        --mode visual
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import training_utils as tu
import util as u
from moco_training import resolve_path, resolve_checkpoint_dir


# ── BYOL components ──────────────────────────────────────────────────────────


class MLPHead(nn.Module):
    """Two-layer MLP with BN (projector or predictor)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Symmetric negative cosine similarity loss.

    p: online predictor output  (B, D)
    z: target projector output  (B, D)  — stop-gradient applied by caller
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


@torch.no_grad()
def _update_ema(online: nn.Module, target: nn.Module, tau: float) -> None:
    """Exponential moving average: target ← tau * target + (1-tau) * online."""
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)


# ── data helpers ─────────────────────────────────────────────────────────────


def _build_loaders(
    config_path: Path,
    data_cfg: Dict[str, object],
    *,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset_type = str(data_cfg.get("dataset_type", "cronos")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))
    normalize = bool(data_cfg.get("normalize", True))
    val_ratio_cfg = data_cfg.get("val_ratio")
    val_ratio = float(val_ratio_cfg) if val_ratio_cfg is not None else 0.2
    train_ratio = float(data_cfg.get("train_ratio", 0.8))

    cronos_kwargs: Dict[str, object] = dict(data_cfg.get("cronos_kwargs", {}) or {})
    datasets_spec = data_cfg.get("datasets")
    dataset_name = data_cfg.get("dataset_name")

    resolved = resolve_path(config_path.parent, data_cfg.get("data_dir"))
    data_root = resolved if resolved is not None else (config_path.parent.parent / "data").resolve()

    if dataset_type == "cronos":
        load_kwargs = cronos_kwargs.setdefault("load_kwargs", {})
        load_kwargs.setdefault("offline_cache_dir", str(data_root))
        load_kwargs.setdefault("force_offline", True)

    train_loader, val_loader = u.build_time_series_dataloaders(
        data_dir=str(data_root),
        filename=data_cfg.get("filename"),
        dataset_name=dataset_name,
        datasets=datasets_spec,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        dataset_type=dataset_type,
        val_split=data_cfg.get("val_split"),
        seed=seed,
        cronos_kwargs=cronos_kwargs,
    )
    return train_loader, val_loader


# ── training loop ────────────────────────────────────────────────────────────


def run_byol_training(
    *,
    encoder: nn.Module,
    projector: nn.Module,
    predictor: nn.Module,
    mode: str,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int = 100,
    noise_std: float = 0.05,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object] = None,
    ema_tau_base: float = 0.996,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    experiment: Optional[object] = None,
) -> None:
    """Main BYOL training loop.

    Online network:  encoder → projector → predictor
    Target network:  encoder_ema → projector_ema  (no grad, EMA updated)
    Loss:            symmetric BYOL loss (no negatives)
    """
    # Build target (EMA) network — deep copy, detached
    encoder_ema = copy.deepcopy(encoder).to(device)
    projector_ema = copy.deepcopy(projector).to(device)
    for p in list(encoder_ema.parameters()) + list(projector_ema.parameters()):
        p.requires_grad_(False)

    encoder.to(device)
    projector.to(device)
    predictor.to(device)

    for state in optimizer.state.values():
        for key, val in list(state.items()):
            if torch.is_tensor(val):
                state[key] = val.to(device)

    best_metric = float("inf") if best_loss is None else float(best_loss)
    start_epoch = max(0, int(initial_epoch))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_steps = (epochs - start_epoch) * len(train_loader)

    def _encode(enc: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return enc(x)

    def _augment(x: torch.Tensor) -> torch.Tensor:
        return x + noise_std * torch.randn_like(x)

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, epochs):
        encoder.train()
        projector.train()
        predictor.train()

        # Cosine schedule for EMA tau: tau increases from tau_base toward 1
        epoch_loss = 0.0
        batches = 0
        desc = f"Epoch {epoch + 1}/{epochs}"

        with tqdm(train_loader, desc=desc) as pbar:
            for batch in pbar:
                # cosine tau schedule per step
                tau = 1.0 - (1.0 - ema_tau_base) * (
                    (1.0 + torch.cos(torch.tensor(torch.pi * global_step / max(total_steps, 1)))).item() / 2.0
                )

                # ── extract raw series ──────────────────────────────────────
                if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
                    padded = batch["target"].to(device).float()
                    lengths = batch["lengths"].to(device)
                    if not (lengths == lengths[0]).all():
                        # variable length — skip for simplicity (rare)
                        global_step += 1
                        continue
                    L = int(lengths[0].item())
                    x_raw = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
                else:
                    seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
                    x_raw = u.reshape_multivariate_series(seq)

                # ── two augmented views ────────────────────────────────────
                x1 = _augment(x_raw)
                x2 = _augment(x_raw)

                # ── online forward ─────────────────────────────────────────
                z1_online = projector(_encode(encoder, x1))
                z2_online = projector(_encode(encoder, x2))
                p1 = predictor(z1_online)
                p2 = predictor(z2_online)

                # ── target forward (no grad) ───────────────────────────────
                with torch.no_grad():
                    z1_target = projector_ema(_encode(encoder_ema, x1))
                    z2_target = projector_ema(_encode(encoder_ema, x2))

                # ── symmetric BYOL loss ────────────────────────────────────
                loss = 0.5 * (byol_loss(p1, z2_target) + byol_loss(p2, z1_target))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # ── EMA update ─────────────────────────────────────────────
                _update_ema(encoder, encoder_ema, tau)
                _update_ema(projector, projector_ema, tau)

                if scheduler is not None:
                    scheduler.step()

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                batches += 1
                global_step += 1
                pbar.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{epoch_loss/batches:.4f}", tau=f"{tau:.4f}")

        train_loss = epoch_loss / batches if batches > 0 else float("nan")

        # ── validation ─────────────────────────────────────────────────────
        val_loss = None
        if val_loader is not None:
            encoder.eval()
            projector.eval()
            predictor.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for val_batch in val_loader:
                    if isinstance(val_batch, dict) and "target" in val_batch and "lengths" in val_batch:
                        padded = val_batch["target"].to(device).float()
                        lengths = val_batch["lengths"].to(device)
                        if not (lengths == lengths[0]).all():
                            continue
                        L = int(lengths[0].item())
                        vx = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
                    else:
                        vseq = u.prepare_sequence(u.extract_sequence(val_batch)).to(device).float()
                        vx = u.reshape_multivariate_series(vseq)

                    vx1 = _augment(vx)
                    vx2 = _augment(vx)
                    vz1 = projector_ema(_encode(encoder_ema, vx1))
                    vz2 = projector_ema(_encode(encoder_ema, vx2))
                    vp1 = predictor(projector(_encode(encoder, vx1)))
                    vp2 = predictor(projector(_encode(encoder, vx2)))
                    val_total += 0.5 * (byol_loss(vp1, vz2) + byol_loss(vp2, vz1)).item()
                    val_batches += 1
            val_loss = val_total / val_batches if val_batches > 0 else None
            encoder.train()
            projector.train()
            predictor.train()

        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            experiment.log_metric("ema_tau", tau, step=epoch + 1)

        monitor_loss = val_loss if val_loss is not None else train_loss
        is_best = monitor_loss < best_metric
        if is_best:
            best_metric = monitor_loss

        ckpt_name = "time_series" if mode == "temporal" else "visual_encoder"

        def _save(suffix: str) -> None:
            path = checkpoint_dir / f"{ckpt_name}_{suffix}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": encoder.state_dict(),
                    "ema_state_dict": encoder_ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": monitor_loss,
                },
                path,
            )
            proj_path = checkpoint_dir / f"{ckpt_name}_projection_{suffix}.pt"
            torch.save({"epoch": epoch, "model_state_dict": projector.state_dict()}, proj_path)

        _save("last")
        if is_best:
            _save("best")

        best_tag = " [best]" if is_best else ""
        val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
        print(f"Epoch {epoch + 1}/{epochs}  train={train_loss:.4f}{val_str}{best_tag}")

    print(f"BYOL training complete. Best loss: {best_metric:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BYOL single-encoder self-supervised training")
    default_cfg = Path(__file__).resolve().parent / "configs" / "lotsa_byol_temporal.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--mode", choices=["temporal", "visual"], default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = tu.load_config(config_path)

    mode_from_config = str(config.model.get("byol_mode", config.model.get("rp_encoder", config.model.get("simclr_mode", "")))).lower()
    if mode_from_config == "upper_tri":
        mode_from_config = "visual"
    if args.mode is not None:
        mode = args.mode
    elif mode_from_config in ("temporal", "visual"):
        mode = mode_from_config
    else:
        raise ValueError("Encoder mode not specified. Pass --mode temporal or --mode visual.")

    from comet_utils import create_comet_experiment
    exp_name = str(getattr(config, "experiment_name", f"byol_{mode}"))
    comet_key_map = {
        "ts_byol_temporal_lotsa": "byol_temporal",
        "ts_byol_visual_lotsa": "byol_visual",
    }
    comet_key = comet_key_map.get(exp_name, f"byol_{mode}")
    experiment = create_comet_experiment(comet_key)

    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Starting BYOL training — mode={mode}  device={device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 100))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.05))
    lr = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    ema_tau = float(training_cfg.get("ema_tau", 0.996))

    experiment.log_parameters({
        "mode": mode, "epochs": epochs, "noise_std": noise_std,
        "learning_rate": lr, "weight_decay": weight_decay,
        "ema_tau": ema_tau, "seed": config.seed,
    })

    train_loader, val_loader = _build_loaders(config_path, config.data, seed=config.seed)
    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    model_cfg = config.model
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    model_dim = int(model_cfg.get("model_dim", 256))
    proj_dim = int(model_cfg.get("proj_dim", 256))
    pred_dim = int(model_cfg.get("pred_dim", 128))

    if mode == "temporal":
        encoder = tu.build_encoder_from_config(model_cfg)
        enc_out_dim = embedding_dim
        print(f"Temporal encoder: MambaEncoder (embedding_dim={enc_out_dim})")
    else:
        from models.mamba_visual_encoder import UpperTriDiagRPEncoder
        encoder = UpperTriDiagRPEncoder(
            patch_len=int(model_cfg.get("input_dim", 32)),
            d_model=model_dim,
            n_layers=int(model_cfg.get("depth", 8)),
            embedding_dim=embedding_dim,
            rp_mv_strategy="mean",
        )
        enc_out_dim = embedding_dim
        print(f"Visual encoder: UpperTriDiagRPEncoder (patch_len={model_cfg.get('input_dim', 32)}, embedding_dim={enc_out_dim})")

    # Online: encoder → projector → predictor
    # Target: encoder_ema → projector_ema  (built inside run_byol_training)
    projector = MLPHead(enc_out_dim, proj_dim, proj_dim)
    predictor = MLPHead(proj_dim, pred_dim, proj_dim)

    params = list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Cosine LR scheduler (per-epoch, not per-step — step() called in loop via scheduler=None)
    # We let the BYOL loop handle LR via optimizer; add a scheduler here if needed
    warmup = int(training_cfg.get("warmup_epochs", 10))
    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return float(ep + 1) / float(warmup)
        progress = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    initial_epoch = 0
    best_loss: Optional[float] = None
    resume_dir: Optional[Path] = None

    if args.resume_checkpoint is not None:
        resume_candidate = resolve_path(Path.cwd(), args.resume_checkpoint)
        if resume_candidate is None:
            raise FileNotFoundError(f"Cannot resolve resume path: {args.resume_checkpoint}")
        resume_dir = resume_candidate if resume_candidate.is_dir() else resume_candidate.parent
        ckpt_name = "time_series" if mode == "temporal" else "visual_encoder"
        last_path = resume_dir / f"{ckpt_name}_last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {last_path}")
        state = torch.load(last_path, map_location="cpu")
        encoder.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = min(epochs, int(state.get("epoch", 0)) + 1)
        best_loss = state.get("loss")
        print(f"Resumed from {last_path} (epoch {initial_epoch})")

    checkpoint_dir = (
        resume_dir.resolve()
        if resume_dir is not None
        else resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    )
    print(f"Config: {config_path.name}")
    print(f"Checkpoints: {checkpoint_dir}")

    run_byol_training(
        encoder=encoder,
        projector=projector,
        predictor=predictor,
        mode=mode,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        optimizer=optimizer,
        ema_tau_base=ema_tau,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
        experiment=experiment,
    )

    # Step LR scheduler after each epoch (called here for cleanliness)
    experiment.end()


if __name__ == "__main__":
    main()
