"""Bimodal SimCLR training: temporal + RP visual branch, NT-Xent multiview loss.

Trains two encoders jointly using NT-Xent (SimCLR) loss over 4 views per sample:
  - z_t1, z_t2 : two noisy temporal views  (MambaEncoder)
  - z_v1, z_v2 : two noisy visual RP views (UpperTriDiagRPEncoder)

Loss:
    L = nt_xent(z_t1, z_t2)                            # intra-temporal
      + nt_xent(z_v1, z_v2)                            # intra-visual
      + 0.5 * (nt_xent(z_t1, z_v1) + nt_xent(z_t2, z_v2))  # cross-modal

Checkpoint layout mirrors cosine_training.py so probe_lotsa_checkpoint.py works
without modification:
    {checkpoint_dir}/time_series_best.pt      — temporal encoder
    {checkpoint_dir}/visual_encoder_best.pt   — visual encoder

Usage
-----
    python3 src/simclr_bimodal_training.py \\
        --config src/configs/lotsa_simclr_bimodal_nano.yaml

    # quick smoke test (2 batches)
    python3 src/simclr_bimodal_training.py \\
        --config src/configs/lotsa_simclr_bimodal_nano.yaml --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import training_utils as tu
import util as u


# ── path + data helpers ───────────────────────────────────────────────────────


def _build_loaders(
    config_path: Path,
    data_cfg: Dict[str, object],
    *,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val dataloaders from config (no moco_training dependency)."""
    dataset_type = str(data_cfg.get("dataset_type", "cronos")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))
    normalize = bool(data_cfg.get("normalize", True))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio_cfg = data_cfg.get("val_ratio")
    val_ratio = float(val_ratio_cfg) if val_ratio_cfg is not None else 0.2

    cronos_kwargs: Dict[str, object] = dict(data_cfg.get("cronos_kwargs", {}) or {})
    datasets_spec = data_cfg.get("datasets")
    dataset_name = data_cfg.get("dataset_name")

    # Resolve data root relative to config location
    data_dir_raw = data_cfg.get("data_dir")
    if data_dir_raw is not None:
        data_root = (config_path.parent / Path(data_dir_raw)).resolve()
    else:
        data_root = (config_path.parent.parent / "data").resolve()

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


from path_utils import resolve_path as _resolve_path, resolve_checkpoint_dir as _resolve_checkpoint_dir


# ── loss ──────────────────────────────────────────────────────────────────────


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric NT-Xent (SimCLR) loss. z1, z2: (N, D) L2-normalised."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def bimodal_simclr_loss(
    z_t1: torch.Tensor,
    z_t2: torch.Tensor,
    z_v1: torch.Tensor,
    z_v2: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Three-term NT-Xent loss over 4 views.

    Returns total loss and a dict of per-term scalars for logging.
    """
    l_tt = _nt_xent(z_t1, z_t2, temperature)
    l_vv = _nt_xent(z_v1, z_v2, temperature)
    l_tv = 0.5 * (_nt_xent(z_t1, z_v1, temperature) + _nt_xent(z_t2, z_v2, temperature))
    total = l_tt + l_vv + l_tv
    parts = {
        "loss_tt": float(l_tt.item()),
        "loss_vv": float(l_vv.item()),
        "loss_tv": float(l_tv.item()),
    }
    return total, parts


# ── projection head ───────────────────────────────────────────────────────────


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR-style)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── batch helpers ─────────────────────────────────────────────────────────────


def _extract_views(
    batch,
    device: torch.device,
    noise_std: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (view1, view2) tensors of shape (N, F, T), or None if batch is skipped."""
    if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
        padded = batch["target"].to(device).float()
        lengths = batch["lengths"].to(device)
        if not (lengths == lengths[0]).all():
            return None
        L = int(lengths[0].item())
        x1 = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
        if "target2" in batch:
            padded2 = batch["target2"].to(device).float()
            x2 = u.reshape_multivariate_series(u.prepare_sequence(padded2[:, :L]))
        else:
            x2 = x1 + noise_std * torch.randn_like(x1)
    else:
        seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
        x1 = u.reshape_multivariate_series(seq)
        x2 = x1 + noise_std * torch.randn_like(x1)
    return x1, x2


# ── training loop ─────────────────────────────────────────────────────────────


def run_bimodal_simclr(
    *,
    ts_encoder: nn.Module,
    rp_encoder: nn.Module,
    ts_proj: ProjectionHead,
    rp_proj: ProjectionHead,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int,
    noise_std: float,
    temperature: float,
    optimizer: torch.optim.Optimizer,
    scheduler,
    use_amp: bool,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    save_best_only: bool = True,
    experiment=None,
    smoke: bool = False,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for m in (ts_encoder, rp_encoder, ts_proj, rp_proj):
        m.to(device)

    scaler = GradScaler(enabled=use_amp)
    best_metric = float("inf") if best_loss is None else float(best_loss)

    def _save(suffix: str, epoch: int, loss: float) -> None:
        for name, model in [("time_series", ts_encoder), ("visual_encoder", rp_encoder),
                             ("time_series_projection", ts_proj), ("visual_projection", rp_proj)]:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": loss},
                checkpoint_dir / f"{name}_{suffix}.pt",
            )

    for epoch in range(initial_epoch, epochs):
        ts_encoder.train(); rp_encoder.train(); ts_proj.train(); rp_proj.train()

        epoch_loss = 0.0
        n_batches = 0
        total = len(train_loader) if hasattr(train_loader, "__len__") else None

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", total=total) as pbar:
            for batch_idx, batch in enumerate(pbar):
                views = _extract_views(batch, device, noise_std)
                if views is None:
                    continue
                x1, x2 = views

                with autocast(enabled=use_amp):
                    z_t1 = ts_proj(ts_encoder(x1))
                    z_t2 = ts_proj(ts_encoder(x2))
                    z_v1 = rp_proj(rp_encoder(x1))
                    z_v2 = rp_proj(rp_encoder(x2))
                    loss, parts = bimodal_simclr_loss(z_t1, z_t2, z_v1, z_v2, temperature)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(ts_encoder.parameters()) + list(rp_encoder.parameters()) +
                    list(ts_proj.parameters()) + list(rp_proj.parameters()),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                n_batches += 1
                pbar.set_postfix(loss=f"{batch_loss:.4f}", tt=f"{parts['loss_tt']:.3f}",
                                 vv=f"{parts['loss_vv']:.3f}", tv=f"{parts['loss_tv']:.3f}")

                if smoke and batch_idx >= 1:
                    print("Smoke test: 2 batches OK")
                    return

        train_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")
        scheduler.step()

        # Validation
        val_loss = None
        if val_loader is not None:
            ts_encoder.eval(); rp_encoder.eval(); ts_proj.eval(); rp_proj.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vviews = _extract_views(vbatch, device, noise_std)
                    if vviews is None:
                        continue
                    vx1, vx2 = vviews
                    with autocast(enabled=use_amp):
                        vz_t1 = ts_proj(ts_encoder(vx1))
                        vz_t2 = ts_proj(ts_encoder(vx2))
                        vz_v1 = rp_proj(rp_encoder(vx1))
                        vz_v2 = rp_proj(rp_encoder(vx2))
                        vloss, _ = bimodal_simclr_loss(vz_t1, vz_t2, vz_v1, vz_v2, temperature)
                    vtotal += float(vloss.item())
                    vn += 1
            val_loss = vtotal / vn if vn > 0 else None
            ts_encoder.train(); rp_encoder.train(); ts_proj.train(); rp_proj.train()

        monitor = val_loss if val_loss is not None else train_loss
        is_best = monitor < best_metric
        if is_best:
            best_metric = monitor

        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            experiment.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch + 1)

        _save("last", epoch, monitor)
        if is_best:
            _save("best", epoch, monitor)

        val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
        best_tag = " [best]" if is_best else ""
        print(f"Epoch {epoch+1}/{epochs}  train={train_loss:.4f}{val_str}{best_tag}")

    print(f"Training complete. Best loss: {best_metric:.4f}  Checkpoints: {checkpoint_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bimodal SimCLR (temporal + RP) training")
    default_cfg = Path(__file__).resolve().parent / "configs" / "lotsa_simclr_bimodal_nano.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run 2 batches and exit (sanity check)")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    config_path = _resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = tu.load_config(config_path)

    from comet_utils import create_comet_experiment
    experiment = create_comet_experiment("simclr_bimodal")

    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Device: {device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 100))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.01))
    temperature = float(training_cfg.get("temperature", 0.2))
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    warmup_epochs = int(training_cfg.get("warmup_epochs", 5))
    use_amp = bool(training_cfg.get("use_amp", True))

    experiment.log_parameters({
        "epochs": epochs, "noise_std": noise_std, "temperature": temperature,
        "learning_rate": lr, "weight_decay": weight_decay, "seed": config.seed,
    })

    train_loader, val_loader = _build_loaders(config_path, config.data, seed=config.seed)
    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    # Build encoders
    ts_encoder = tu.build_encoder_from_config(config.model)
    rp_enc = tu.build_visual_encoder_from_config(config.model)

    emb_dim = int(config.model.get("embedding_dim", 64))
    proj_hidden = int(config.model.get("model_dim", 128))

    ts_proj = ProjectionHead(emb_dim, proj_hidden, emb_dim)
    rp_proj = ProjectionHead(emb_dim, proj_hidden, emb_dim)

    print(f"Temporal encoder params: {sum(p.numel() for p in ts_encoder.parameters()):,}")
    print(f"Visual encoder params:   {sum(p.numel() for p in rp_enc.parameters()):,}")

    params = (
        list(ts_encoder.parameters()) + list(rp_enc.parameters()) +
        list(ts_proj.parameters()) + list(rp_proj.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Warmup + cosine schedule
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        return 1.0
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)

    class _CombinedScheduler:
        def step(self) -> None:
            nonlocal _epoch_counter
            _epoch_counter += 1
            if _epoch_counter <= warmup_epochs:
                warmup_sched.step()
            else:
                cosine_sched.step()

    _epoch_counter = 0
    scheduler = _CombinedScheduler()

    # Resolve checkpoint dir
    initial_epoch = 0
    best_loss = None
    resume_dir: Optional[Path] = None

    if args.resume_checkpoint is not None:
        resume_dir = Path(args.resume_checkpoint).resolve()
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume dir not found: {resume_dir}")
        for name, model in [("time_series", ts_encoder), ("visual_encoder", rp_enc),
                             ("time_series_projection", ts_proj), ("visual_projection", rp_proj)]:
            p = resume_dir / f"{name}_last.pt"
            if p.exists():
                state = torch.load(p, map_location="cpu")
                model.load_state_dict(state["model_state_dict"])
        ts_state = torch.load(resume_dir / "time_series_last.pt", map_location="cpu")
        optimizer.load_state_dict(ts_state["optimizer_state_dict"])
        initial_epoch = int(ts_state.get("epoch", 0)) + 1
        best_loss = ts_state.get("loss")
        print(f"Resumed from {resume_dir} (epoch {initial_epoch})")

    checkpoint_dir = (
        resume_dir if resume_dir is not None
        else _resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    )
    print(f"Checkpoints → {checkpoint_dir}")

    run_bimodal_simclr(
        ts_encoder=ts_encoder,
        rp_encoder=rp_enc,
        ts_proj=ts_proj,
        rp_proj=rp_proj,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        temperature=temperature,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=use_amp,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
        experiment=experiment,
        smoke=args.smoke,
    )

    experiment.end()


if __name__ == "__main__":
    main()
