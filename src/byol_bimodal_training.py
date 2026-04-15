"""BYOL Bimodal training: temporal + RP visual branch with EMA target networks.

Extends the unimodal BYOL to a bimodal setting:
  - Online temporal:  MambaEncoder → projector_t → predictor_t
  - Online visual:    UpperTriDiagRPEncoder → projector_v → predictor_v
  - Target temporal:  MambaEncoder_ema → projector_t_ema  (EMA, no grad)
  - Target visual:    UpperTriDiagRPEncoder_ema → projector_v_ema  (EMA, no grad)

Loss (4 terms):
    L_tt = byol(p_t1 → z_t2_ema) + byol(p_t2 → z_t1_ema)   # intra-temporal BYOL
    L_vv = byol(p_v1 → z_v2_ema) + byol(p_v2 → z_v1_ema)   # intra-visual BYOL
    L_tv = byol(p_t1 → z_v1_ema) + byol(p_t2 → z_v2_ema)   # cross-modal: temporal → visual
    L_vt = byol(p_v1 → z_t1_ema) + byol(p_v2 → z_t2_ema)   # cross-modal: visual → temporal
    L = L_tt + L_vv + 0.5 * (L_tv + L_vt)

Key design decisions:
  - EMA tau = 0.996 (cosine-annealed per step, same as unimodal BYOL)
  - noise_std = 0.05 (stronger than SimCLR 0.01 — consistent with BYOL best practice)
  - No negative pairs → stable with small batches
  - Checkpoint layout mirrors cosine_training.py so probe_lotsa_checkpoint.py works:
      {checkpoint_dir}/time_series_best.pt      — temporal encoder (online)
      {checkpoint_dir}/visual_encoder_best.pt   — visual encoder (online)

Usage
-----
    python3 src/byol_bimodal_training.py \\
        --config src/configs/lotsa_byol_bimodal_nano.yaml

    # smoke test (2 batches)
    python3 src/byol_bimodal_training.py \\
        --config src/configs/lotsa_byol_bimodal_nano.yaml --smoke
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import training_utils as tu
import util as u
from path_utils import resolve_path, resolve_checkpoint_dir


# ── BYOL components ───────────────────────────────────────────────────────────


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
    """Negative cosine similarity (symmetric BYOL loss).

    p: online predictor output  (B, D)
    z: target projector output  (B, D)  — stop-gradient applied by caller
    Returns scalar loss.
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2.0 - 2.0 * (p * z).sum(dim=-1).mean()


@torch.no_grad()
def _update_ema(online: nn.Module, target: nn.Module, tau: float) -> None:
    """EMA: target ← tau * target + (1 - tau) * online."""
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)


# ── data helpers ──────────────────────────────────────────────────────────────


def _build_loaders(
    config_path: Path,
    data_cfg: Dict,
    *,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    dataset_type = str(data_cfg.get("dataset_type", "cronos")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))
    normalize = bool(data_cfg.get("normalize", True))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio_cfg = data_cfg.get("val_ratio")
    val_ratio = float(val_ratio_cfg) if val_ratio_cfg is not None else 0.2

    cronos_kwargs: Dict = dict(data_cfg.get("cronos_kwargs", {}) or {})
    datasets_spec = data_cfg.get("datasets")
    dataset_name = data_cfg.get("dataset_name")

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


def _extract_views(
    batch,
    device: torch.device,
    noise_std: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (view1, view2) tensors (N, F, T), or None to skip batch."""
    if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
        padded = batch["target"].to(device).float()
        lengths = batch["lengths"].to(device)
        if not (lengths == lengths[0]).all():
            return None
        L = int(lengths[0].item())
        x1 = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
        if "target2" in batch:
            x2 = u.reshape_multivariate_series(
                u.prepare_sequence(batch["target2"].to(device).float()[:, :L])
            )
        else:
            x2 = x1 + noise_std * torch.randn_like(x1)
    else:
        seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
        x1 = u.reshape_multivariate_series(seq)
        x2 = x1 + noise_std * torch.randn_like(x1)
    return x1, x2


# ── training loop ─────────────────────────────────────────────────────────────


def run_byol_bimodal(
    *,
    ts_encoder: nn.Module,
    rp_encoder: nn.Module,
    ts_proj: MLPHead,
    rp_proj: MLPHead,
    ts_pred: MLPHead,
    rp_pred: MLPHead,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int,
    noise_std: float,
    ema_tau_base: float,
    optimizer: torch.optim.Optimizer,
    scheduler,
    use_amp: bool,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    experiment=None,
    smoke: bool = False,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build EMA (target) networks
    ts_enc_ema = copy.deepcopy(ts_encoder).to(device)
    rp_enc_ema = copy.deepcopy(rp_encoder).to(device)
    ts_proj_ema = copy.deepcopy(ts_proj).to(device)
    rp_proj_ema = copy.deepcopy(rp_proj).to(device)
    for p in (
        list(ts_enc_ema.parameters()) + list(rp_enc_ema.parameters()) +
        list(ts_proj_ema.parameters()) + list(rp_proj_ema.parameters())
    ):
        p.requires_grad_(False)

    for m in (ts_encoder, rp_encoder, ts_proj, rp_proj, ts_pred, rp_pred):
        m.to(device)

    scaler = GradScaler(enabled=use_amp)
    best_metric = float("inf") if best_loss is None else float(best_loss)

    total_steps = (epochs - initial_epoch) * len(train_loader)
    global_step = initial_epoch * len(train_loader)

    def _save(suffix: str, epoch: int, loss: float) -> None:
        for name, model in [
            ("time_series", ts_encoder),
            ("visual_encoder", rp_encoder),
            ("time_series_projection", ts_proj),
            ("visual_projection", rp_proj),
        ]:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": loss},
                checkpoint_dir / f"{name}_{suffix}.pt",
            )

    for epoch in range(initial_epoch, epochs):
        for m in (ts_encoder, rp_encoder, ts_proj, rp_proj, ts_pred, rp_pred):
            m.train()

        epoch_loss = 0.0
        n_batches = 0
        total = len(train_loader) if hasattr(train_loader, "__len__") else None

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", total=total) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Cosine tau schedule per step
                tau = 1.0 - (1.0 - ema_tau_base) * (
                    (1.0 + torch.cos(torch.tensor(torch.pi * global_step / max(total_steps, 1)))).item() / 2.0
                )

                views = _extract_views(batch, device, noise_std)
                if views is None:
                    global_step += 1
                    continue
                x1, x2 = views

                with autocast(enabled=use_amp):
                    # Online forward
                    z_t1 = ts_proj(ts_encoder(x1))
                    z_t2 = ts_proj(ts_encoder(x2))
                    z_v1 = rp_proj(rp_encoder(x1))
                    z_v2 = rp_proj(rp_encoder(x2))
                    p_t1 = ts_pred(z_t1)
                    p_t2 = ts_pred(z_t2)
                    p_v1 = rp_pred(z_v1)
                    p_v2 = rp_pred(z_v2)

                    # Target forward (no grad)
                    with torch.no_grad():
                        zt1_ema = ts_proj_ema(ts_enc_ema(x1))
                        zt2_ema = ts_proj_ema(ts_enc_ema(x2))
                        zv1_ema = rp_proj_ema(rp_enc_ema(x1))
                        zv2_ema = rp_proj_ema(rp_enc_ema(x2))

                    # BYOL loss terms
                    l_tt = 0.5 * (byol_loss(p_t1, zt2_ema) + byol_loss(p_t2, zt1_ema))
                    l_vv = 0.5 * (byol_loss(p_v1, zv2_ema) + byol_loss(p_v2, zv1_ema))
                    l_tv = 0.5 * (byol_loss(p_t1, zv1_ema) + byol_loss(p_t2, zv2_ema))
                    l_vt = 0.5 * (byol_loss(p_v1, zt1_ema) + byol_loss(p_v2, zt2_ema))
                    loss = l_tt + l_vv + 0.5 * (l_tv + l_vt)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(ts_encoder.parameters()) + list(rp_encoder.parameters()) +
                    list(ts_proj.parameters()) + list(rp_proj.parameters()) +
                    list(ts_pred.parameters()) + list(rp_pred.parameters()),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                # EMA update for all target modules
                _update_ema(ts_encoder, ts_enc_ema, tau)
                _update_ema(rp_encoder, rp_enc_ema, tau)
                _update_ema(ts_proj, ts_proj_ema, tau)
                _update_ema(rp_proj, rp_proj_ema, tau)

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                n_batches += 1
                global_step += 1
                pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    tt=f"{float(l_tt):.3f}",
                    vv=f"{float(l_vv):.3f}",
                    tv=f"{0.5*(float(l_tv)+float(l_vt)):.3f}",
                    tau=f"{tau:.4f}",
                )

                if smoke and batch_idx >= 1:
                    print("Smoke test: 2 batches OK")
                    return

        train_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")
        scheduler.step()

        # Validation (using EMA target networks for stability)
        val_loss = None
        if val_loader is not None:
            for m in (ts_encoder, rp_encoder, ts_proj, rp_proj, ts_pred, rp_pred):
                m.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vviews = _extract_views(vbatch, device, noise_std)
                    if vviews is None:
                        continue
                    vx1, vx2 = vviews
                    with autocast(enabled=use_amp):
                        vzt1 = ts_proj_ema(ts_enc_ema(vx1))
                        vzt2 = ts_proj_ema(ts_enc_ema(vx2))
                        vzv1 = rp_proj_ema(rp_enc_ema(vx1))
                        vzv2 = rp_proj_ema(rp_enc_ema(vx2))
                        vp_t1 = ts_pred(ts_proj(ts_encoder(vx1)))
                        vp_t2 = ts_pred(ts_proj(ts_encoder(vx2)))
                        vp_v1 = rp_pred(rp_proj(rp_encoder(vx1)))
                        vp_v2 = rp_pred(rp_proj(rp_encoder(vx2)))
                        vl_tt = 0.5 * (byol_loss(vp_t1, vzt2) + byol_loss(vp_t2, vzt1))
                        vl_vv = 0.5 * (byol_loss(vp_v1, vzv2) + byol_loss(vp_v2, vzv1))
                        vl_tv = 0.5 * (byol_loss(vp_t1, vzv1) + byol_loss(vp_t2, vzv2))
                        vl_vt = 0.5 * (byol_loss(vp_v1, vzt1) + byol_loss(vp_v2, vzt2))
                        vloss = vl_tt + vl_vv + 0.5 * (vl_tv + vl_vt)
                    vtotal += float(vloss.item())
                    vn += 1
            val_loss = vtotal / vn if vn > 0 else None
            for m in (ts_encoder, rp_encoder, ts_proj, rp_proj, ts_pred, rp_pred):
                m.train()

        monitor = val_loss if val_loss is not None else train_loss
        is_best = monitor < best_metric
        if is_best:
            best_metric = monitor

        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            experiment.log_metric("ema_tau", tau, step=epoch + 1)
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
    parser = argparse.ArgumentParser(description="BYOL Bimodal (temporal + RP) training")
    default_cfg = Path(__file__).resolve().parent / "configs" / "lotsa_byol_bimodal_nano.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run 2 batches and exit (sanity check)")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = tu.load_config(config_path)

    from comet_utils import create_comet_experiment
    experiment = create_comet_experiment("byol_bimodal")

    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Device: {device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 100))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.05))
    ema_tau = float(training_cfg.get("ema_tau", 0.996))
    lr = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    warmup_epochs = int(training_cfg.get("warmup_epochs", 10))
    use_amp = bool(training_cfg.get("use_amp", True))

    experiment.log_parameters({
        "epochs": epochs, "noise_std": noise_std, "ema_tau": ema_tau,
        "learning_rate": lr, "weight_decay": weight_decay, "seed": config.seed,
    })

    train_loader, val_loader = _build_loaders(config_path, config.data, seed=config.seed)
    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    model_cfg = config.model
    emb_dim = int(model_cfg.get("embedding_dim", 64))
    proj_dim = int(model_cfg.get("proj_dim", emb_dim * 2))
    pred_dim = int(model_cfg.get("pred_dim", emb_dim))

    ts_encoder = tu.build_encoder_from_config(model_cfg)
    rp_encoder = tu.build_visual_encoder_from_config(model_cfg)

    ts_proj = MLPHead(emb_dim, proj_dim, proj_dim)
    rp_proj = MLPHead(emb_dim, proj_dim, proj_dim)
    ts_pred = MLPHead(proj_dim, pred_dim, proj_dim)
    rp_pred = MLPHead(proj_dim, pred_dim, proj_dim)

    print(f"Temporal encoder params: {sum(p.numel() for p in ts_encoder.parameters()):,}")
    print(f"Visual encoder params:   {sum(p.numel() for p in rp_encoder.parameters()):,}")

    params = (
        list(ts_encoder.parameters()) + list(rp_encoder.parameters()) +
        list(ts_proj.parameters()) + list(rp_proj.parameters()) +
        list(ts_pred.parameters()) + list(rp_pred.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Warmup + cosine LR schedule
    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        return 1.0

    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
    warmup_sched = LambdaLR(optimizer, _lr_lambda)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)

    class _CombinedScheduler:
        def __init__(self) -> None:
            self._ep = 0

        def step(self) -> None:
            self._ep += 1
            if self._ep <= warmup_epochs:
                warmup_sched.step()
            else:
                cosine_sched.step()

    scheduler = _CombinedScheduler()

    initial_epoch = 0
    best_loss: Optional[float] = None

    if args.resume_checkpoint is not None:
        resume_dir = Path(args.resume_checkpoint).resolve()
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume dir not found: {resume_dir}")
        for name, model in [
            ("time_series", ts_encoder),
            ("visual_encoder", rp_encoder),
            ("time_series_projection", ts_proj),
            ("visual_projection", rp_proj),
        ]:
            p = resume_dir / f"{name}_last.pt"
            if p.exists():
                model.load_state_dict(torch.load(p, map_location="cpu")["model_state_dict"])
        ts_state = torch.load(resume_dir / "time_series_last.pt", map_location="cpu")
        optimizer.load_state_dict(ts_state["optimizer_state_dict"])
        initial_epoch = int(ts_state.get("epoch", 0)) + 1
        best_loss = ts_state.get("loss")
        checkpoint_dir = resume_dir
        print(f"Resumed from {resume_dir} (epoch {initial_epoch})")
    else:
        checkpoint_dir = resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)

    print(f"Checkpoints → {checkpoint_dir}")

    run_byol_bimodal(
        ts_encoder=ts_encoder,
        rp_encoder=rp_encoder,
        ts_proj=ts_proj,
        rp_proj=rp_proj,
        ts_pred=ts_pred,
        rp_pred=rp_pred,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        ema_tau_base=ema_tau,
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
