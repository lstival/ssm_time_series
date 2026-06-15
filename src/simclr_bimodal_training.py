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


def _augment_view(
    x: torch.Tensor,
    *,
    noise_std: float,
    scale_std: float = 0.0,
    mask_ratio: float = 0.0,
) -> torch.Tensor:
    """Stronger time-series augmentation for contrastive SSL.

    In contrastive SSL the augmentation IS the main regularizer — weak views make
    the task trivial and the encoder memorizes (the overfit we measured). Applies,
    per call (so the two views differ):
      - jitter:    additive Gaussian noise (noise_std)
      - scaling:   per-series magnitude scale ~ N(1, scale_std)  (amplitude invariance)
      - masking:   randomly zero a fraction of timesteps (temporal dropout)
    x: (N, F, T). All ops are no-ops when their strength is 0 (backward-compatible).
    """
    out = x
    if scale_std > 0.0:
        # one scale per (sample, channel), broadcast over time
        scale = 1.0 + scale_std * torch.randn(out.shape[0], out.shape[1], 1, device=out.device)
        out = out * scale
    if noise_std > 0.0:
        out = out + noise_std * torch.randn_like(out)
    if mask_ratio > 0.0:
        keep = (torch.rand(out.shape[0], 1, out.shape[2], device=out.device) >= mask_ratio).float()
        out = out * keep
    return out


def _extract_views(
    batch,
    device: torch.device,
    noise_std: float,
    *,
    scale_std: float = 0.0,
    mask_ratio: float = 0.0,
    temporal_positive: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (view1, view2) tensors of shape (N, F, T), or None if batch is skipped.

    temporal_positive=True (variants A/G): view1=context window, view2=adjacent future
    window. The two windows come from the same series but different positions, forcing
    the encoder to align representations of temporally adjacent patches — a much stronger
    forecast signal than augmentation of the same window.
    """
    aug = dict(noise_std=noise_std, scale_std=scale_std, mask_ratio=mask_ratio)
    if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
        padded = batch["target"].to(device).float()
        lengths = batch["lengths"].to(device)
        if not (lengths == lengths[0]).all():
            return None
        L = int(lengths[0].item())
        full = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
        if temporal_positive and full.shape[-1] >= 2:
            # split in half: first half = context view, second half = future view
            half = full.shape[-1] // 2
            base1 = full[..., :half]
            base2 = full[..., half:half * 2]
        else:
            base1 = full
            if "target2" in batch:
                padded2 = batch["target2"].to(device).float()
                base2 = u.reshape_multivariate_series(u.prepare_sequence(padded2[:, :L]))
            else:
                base2 = base1
    else:
        seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
        full = u.reshape_multivariate_series(seq)
        if temporal_positive and full.shape[-1] >= 2:
            half = full.shape[-1] // 2
            base1 = full[..., :half]
            base2 = full[..., half:half * 2]
        else:
            base1 = full
            base2 = full
    x1 = _augment_view(base1, **aug)
    x2 = _augment_view(base2, **aug)
    return x1, x2


def _forecast_reg_loss(
    ts_encoder: nn.Module,
    x: torch.Tensor,
    forecast_reg_head: nn.Module,
    horizon: int = 96,
) -> torch.Tensor:
    """Forecast regularization (variant C): MSE between Linear(z_ctx) and x_fut.

    x: (N, 1, T) where T >= 2*horizon. Splits x into context and target, embeds
    the context, predicts the target with a frozen-during-SSL linear head.
    The head and this loss are trained jointly with the contrastive loss.
    """
    T = x.shape[-1]
    ctx_len = T - horizon
    if ctx_len < horizon:
        return torch.tensor(0.0, device=x.device)
    x_ctx = x[..., :ctx_len]   # (N, 1, ctx_len)
    x_fut = x[..., ctx_len:ctx_len + horizon].squeeze(1)   # (N, horizon)
    z = ts_encoder(x_ctx)      # (N, D) — uses current pooling
    pred = forecast_reg_head(z)  # (N, horizon)
    return F.mse_loss(pred, x_fut)


def _masked_recon_loss(
    ts_encoder: nn.Module,
    x: torch.Tensor,
    recon_head: nn.Module,
    mask_ratio: float = 0.25,
) -> torch.Tensor:
    """Masked reconstruction auxiliary loss — temporal branch only (SimMTM-style).

    Zeroes a random fraction of timesteps, embeds the masked series with the
    temporal encoder, and reconstructs the series via Linear(emb_dim → T).
    MSE is computed ONLY on masked positions, forcing the embedding to retain
    local dynamics that NT-Xent alone can discard. The visual/RP branch is
    untouched, keeping it fully dedicated to cross-modal alignment.
    x: (N, F, T); reconstruction targets channel 0, matching the probe.
    """
    T = x.shape[-1]
    if T > recon_head.out_features:
        return torch.tensor(0.0, device=x.device)
    mask = torch.rand(x.shape[0], 1, T, device=x.device) < mask_ratio  # True = masked
    if not mask.any():
        return torch.tensor(0.0, device=x.device)
    x_masked = x * (~mask).float()
    z = ts_encoder(x_masked)                 # (N, D)
    pred = recon_head(z)[:, :T]              # (N, T)
    target = x[:, 0, :]                      # (N, T)
    m = mask[:, 0, :]
    return F.mse_loss(pred[m], target[m])


# ── training loop ─────────────────────────────────────────────────────────────


def _build_forecast_probe(
    ts_encoder: nn.Module,
    rp_encoder: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    horizons: Tuple[int, ...] = (96, 192, 336, 720),
    probe_epochs: int = 5,
    use_amp: bool = True,
) -> Optional[float]:
    """Train a lightweight linear probe on val embeddings and return avg NRMSE.

    Embeds all val batches with the frozen encoders, trains a Linear(emb→H)
    per horizon for probe_epochs, then evaluates MSE. Returns avg NRMSE across
    horizons (lower = better encoder for forecast). Used as an alternative
    checkpoint criterion to the contrastive val loss.
    """
    ts_encoder.eval(); rp_encoder.eval()
    emb_dim_t = None
    emb_dim_v = None
    Zs, Xs = [], []

    with torch.no_grad():
        for batch in val_loader:
            views = _extract_views(batch, device, noise_std=0.0)
            if views is None:
                continue
            x, _ = views
            with autocast(enabled=use_amp):
                zt = ts_encoder(x)
                zv = rp_encoder(x)
            z = torch.cat([zt, zv], dim=-1)
            Zs.append(z.cpu())
            # raw series for target: take last max(horizons) steps of x (B, 1, T)
            raw = x[:, 0, :].cpu()   # (B, T)
            Xs.append(raw)
            if emb_dim_t is None:
                emb_dim_t = zt.shape[-1]
                emb_dim_v = zv.shape[-1]

    if not Zs:
        return None

    Z_all = torch.cat(Zs, dim=0).float()   # (N, D) — cast to fp32 (AMP may produce fp16)
    X_all = torch.cat(Xs, dim=0)           # (N, T)
    T = X_all.shape[1]
    emb_dim = Z_all.shape[1]

    nrmse_list = []
    for H in horizons:
        if H >= T:
            continue
        ctx_len = T - H
        X_ctx = X_all[:, :ctx_len]   # (N, ctx_len) — not used directly (emb is used)
        Y = X_all[:, ctx_len:]       # (N, H) — target

        head = nn.Linear(emb_dim, H).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        Z_dev = Z_all.to(device)
        Y_dev = Y.to(device)

        for _ in range(probe_epochs):
            pred = head(Z_dev)
            loss = F.mse_loss(pred, Y_dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            pred = head(Z_dev)
            rmse = ((pred - Y_dev) ** 2).mean().sqrt()
            scale = Y_dev.abs().mean().clamp(min=1e-8)
            nrmse_list.append((rmse / scale).item())

        del head, Z_dev, Y_dev

    ts_encoder.train(); rp_encoder.train()
    return float(sum(nrmse_list) / len(nrmse_list)) if nrmse_list else None


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
    max_grad_norm: float = 1.0,
    aug_scale_std: float = 0.0,
    aug_mask_ratio: float = 0.0,
    optimizer: torch.optim.Optimizer,
    scheduler,
    warmup_sched=None,
    cosine_sched=None,
    get_epoch_counter=None,
    use_amp: bool = True,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    early_stopping_patience: int = 0,
    forecast_probe_epochs: int = 0,
    save_best_only: bool = True,
    experiment=None,
    smoke: bool = False,
    # Encoder improvement variants
    forecast_reg_lambda: float = 0.0,   # C: weight of forecast MSE regularization in SSL loss
    temporal_positive: bool = False,    # A/G: use adjacent windows as positives
    recon_lambda: float = 0.0,          # masked reconstruction aux loss (temporal branch only)
    recon_mask_ratio: float = 0.25,     # fraction of timesteps masked for reconstruction
    recon_len: int = 512,               # max series length the recon head can output
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for m in (ts_encoder, rp_encoder, ts_proj, rp_proj):
        m.to(device)

    # Variant C: forecast regularization head — Linear(emb_dim → 96), trained jointly
    forecast_reg_head: Optional[nn.Module] = None
    if forecast_reg_lambda > 0.0:
        emb_dim = getattr(ts_encoder, "embedding_dim", 96)
        forecast_reg_head = nn.Linear(emb_dim, 96).to(device)
        optimizer.add_param_group({"params": forecast_reg_head.parameters()})
        print(f"Forecast reg: lambda={forecast_reg_lambda}, Linear({emb_dim}→96) added to optimizer")

    # Masked reconstruction head — temporal branch only, discarded after SSL.
    recon_head: Optional[nn.Module] = None
    if recon_lambda > 0.0:
        emb_dim = getattr(ts_encoder, "embedding_dim", 96)
        recon_head = nn.Linear(emb_dim, recon_len).to(device)
        optimizer.add_param_group({"params": recon_head.parameters()})
        print(f"Masked recon: lambda={recon_lambda}, mask_ratio={recon_mask_ratio}, "
              f"Linear({emb_dim}→{recon_len}) added to optimizer")

    # When resuming, optimizer state (exp_avg/exp_avg_sq) was loaded on CPU while
    # params are now on `device`. Move the optimizer state to match, otherwise the
    # first step() raises "tensors on cuda:0 and cpu".
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    scaler = GradScaler(enabled=use_amp)
    best_metric = float("inf") if best_loss is None else float(best_loss)
    _no_improve = 0

    def _save(suffix: str, epoch: int, loss: float) -> None:
        save_dict = {
            "epoch": epoch,
            "loss": loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "warmup_sched_state": warmup_sched.state_dict() if warmup_sched is not None else None,
            "cosine_sched_state": cosine_sched.state_dict() if cosine_sched is not None else None,
            "epoch_counter": get_epoch_counter() if get_epoch_counter is not None else 0,
        }
        for name, model in [("time_series", ts_encoder), ("visual_encoder", rp_encoder),
                             ("time_series_projection", ts_proj), ("visual_projection", rp_proj)]:
            d = save_dict.copy()
            d["model_state_dict"] = model.state_dict()
            torch.save(d, checkpoint_dir / f"{name}_{suffix}.pt")

    for epoch in range(initial_epoch, epochs):
        ts_encoder.train(); rp_encoder.train(); ts_proj.train(); rp_proj.train()

        epoch_loss = 0.0
        n_batches = 0
        grad_norm_sum = 0.0      # accumulates pre-clip global grad norm
        grad_norm_max = 0.0
        grad_clip_hits = 0       # batches where pre-clip norm exceeded max_norm
        GRAD_CLIP_MAX = max_grad_norm
        total = len(train_loader) if hasattr(train_loader, "__len__") else None

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", total=total) as pbar:
            for batch_idx, batch in enumerate(pbar):
                views = _extract_views(batch, device, noise_std,
                                       scale_std=aug_scale_std, mask_ratio=aug_mask_ratio,
                                       temporal_positive=temporal_positive)
                if views is None:
                    continue
                x1, x2 = views

                with autocast(enabled=use_amp):
                    z_t1 = ts_proj(ts_encoder(x1))
                    z_t2 = ts_proj(ts_encoder(x2))
                    z_v1 = rp_proj(rp_encoder(x1))
                    z_v2 = rp_proj(rp_encoder(x2))
                    loss, parts = bimodal_simclr_loss(z_t1, z_t2, z_v1, z_v2, temperature)

                    # Variant C: add forecast regularization to SSL loss
                    if forecast_reg_lambda > 0.0 and forecast_reg_head is not None:
                        freg = _forecast_reg_loss(ts_encoder, x1, forecast_reg_head, horizon=96)
                        loss = loss + forecast_reg_lambda * freg
                        parts["loss_freg"] = float(freg.item())

                    # Masked reconstruction aux loss — temporal branch only.
                    if recon_lambda > 0.0 and recon_head is not None:
                        lrec = _masked_recon_loss(ts_encoder, x1, recon_head,
                                                  mask_ratio=recon_mask_ratio)
                        loss = loss + recon_lambda * lrec
                        parts["loss_recon"] = float(lrec.item())

                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # clip_grad_norm_ returns the TOTAL pre-clip grad norm — capture it
                # so we can see the real gradient scale instead of clipping blind.
                total_norm = torch.nn.utils.clip_grad_norm_(
                    list(ts_encoder.parameters()) + list(rp_encoder.parameters()) +
                    list(ts_proj.parameters()) + list(rp_proj.parameters()),
                    max_norm=GRAD_CLIP_MAX,
                )
                scaler.step(optimizer)
                scaler.update()

                gn = float(total_norm)
                if gn == gn:  # not NaN
                    grad_norm_sum += gn
                    grad_norm_max = max(grad_norm_max, gn)
                    if gn > GRAD_CLIP_MAX:
                        grad_clip_hits += 1

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                n_batches += 1
                pbar.set_postfix(loss=f"{batch_loss:.4f}", gn=f"{gn:.2f}",
                                 tt=f"{parts['loss_tt']:.3f}",
                                 vv=f"{parts['loss_vv']:.3f}", tv=f"{parts['loss_tv']:.3f}")

                if smoke and batch_idx >= 1:
                    print("Smoke test: 2 batches OK")
                    return

        train_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")
        grad_norm_mean = grad_norm_sum / n_batches if n_batches > 0 else float("nan")
        grad_clip_rate = grad_clip_hits / n_batches if n_batches > 0 else float("nan")
        print(f"  [grad] mean_norm={grad_norm_mean:.3f}  max_norm={grad_norm_max:.3f}  "
              f"clip_hit_rate={grad_clip_rate:.1%} (clip@{GRAD_CLIP_MAX})")
        scheduler.step()

        # Validation
        val_loss = None
        if val_loader is not None:
            ts_encoder.eval(); rp_encoder.eval(); ts_proj.eval(); rp_proj.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vviews = _extract_views(vbatch, device, noise_std,
                                            scale_std=aug_scale_std, mask_ratio=aug_mask_ratio)
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

        # Optional forecast probe — linear head trained on val embeddings.
        # Tracks a forecast-relevant signal independently of contrastive val loss.
        probe_nrmse = None
        if forecast_probe_epochs > 0 and val_loader is not None:
            probe_nrmse = _build_forecast_probe(
                ts_encoder, rp_encoder, val_loader, device,
                probe_epochs=forecast_probe_epochs, use_amp=use_amp,
            )

        monitor = val_loss if val_loss is not None else train_loss
        is_best = monitor < best_metric
        if is_best:
            best_metric = monitor
            _no_improve = 0
        else:
            _no_improve += 1

        # Separately track best checkpoint by probe NRMSE (saved as *_best_probe.pt)
        if probe_nrmse is not None:
            if not hasattr(run_bimodal_simclr, "_best_probe_nrmse"):
                run_bimodal_simclr._best_probe_nrmse = float("inf")
            if probe_nrmse < run_bimodal_simclr._best_probe_nrmse:
                run_bimodal_simclr._best_probe_nrmse = probe_nrmse
                _save("best_probe", epoch, probe_nrmse)

        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            if probe_nrmse is not None:
                experiment.log_metric("probe_nrmse", probe_nrmse, step=epoch + 1)
            experiment.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch + 1)
            experiment.log_metric("grad_norm_mean", grad_norm_mean, step=epoch + 1)
            experiment.log_metric("grad_norm_max", grad_norm_max, step=epoch + 1)
            experiment.log_metric("grad_clip_rate", grad_clip_rate, step=epoch + 1)
            experiment.log_metric("no_improve_epochs", _no_improve, step=epoch + 1)

        _save("last", epoch, monitor)
        if is_best:
            _save("best", epoch, monitor)

        val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
        probe_str = f"  probe_nrmse={probe_nrmse:.4f}" if probe_nrmse is not None else ""
        best_tag = " [best]" if is_best else ""
        es_str = f"  [no-improve {_no_improve}/{early_stopping_patience}]" if early_stopping_patience > 0 and not is_best else ""
        print(f"Epoch {epoch+1}/{epochs}  train={train_loss:.4f}{val_str}{probe_str}{best_tag}{es_str}")

        if early_stopping_patience > 0 and _no_improve >= early_stopping_patience:
            print(f"Early stopping: val did not improve for {early_stopping_patience} epochs.")
            break

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
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    aug_scale_std = float(training_cfg.get("aug_scale_std", 0.0))
    aug_mask_ratio = float(training_cfg.get("aug_mask_ratio", 0.0))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))
    forecast_probe_epochs = int(training_cfg.get("forecast_probe_epochs", 0))
    forecast_reg_lambda = float(training_cfg.get("forecast_reg_lambda", 0.0))
    temporal_positive = bool(training_cfg.get("temporal_positive", False))
    recon_lambda = float(training_cfg.get("recon_lambda", 0.0))
    recon_mask_ratio = float(training_cfg.get("recon_mask_ratio", 0.25))

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
    proj_out_dim = int(config.model.get("proj_out_dim", emb_dim))

    # Ablation: use_projection=false applies the contrastive loss DIRECTLY on the
    # encoder features instead of through a projection head. The projection head
    # normally shields the encoder from the contrastive loss (SimCLR/CLIP design);
    # the forecast (MoP) already consumes raw encoder features, so aligning on them
    # directly may help downstream. Identity keeps the ts_proj(ts_encoder(x)) call
    # unchanged. Default true = baseline.
    use_projection = bool(config.model.get("use_projection", True))
    if use_projection:
        ts_proj = ProjectionHead(emb_dim, proj_hidden, proj_out_dim)
        rp_proj = ProjectionHead(emb_dim, proj_hidden, proj_out_dim)
    else:
        print("Ablation: use_projection=False — contrastive loss on raw encoder features")
        ts_proj = nn.Identity()
        rp_proj = nn.Identity()

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
        def _load_with_remap(model, state_dict):
            # Remap legacy key prefix 'projection.' → 'net.' for ProjectionHead
            remapped = {
                k.replace("projection.", "net.", 1) if k.startswith("projection.") else k: v
                for k, v in state_dict.items()
            }
            model.load_state_dict(remapped)

        for name, model in [("time_series", ts_encoder), ("visual_encoder", rp_enc),
                             ("time_series_projection", ts_proj), ("visual_projection", rp_proj)]:
            p = resume_dir / f"{name}_last.pt"
            if p.exists():
                state = torch.load(p, map_location="cpu")
                _load_with_remap(model, state["model_state_dict"])
        
        # Load optimizer and scheduler if present
        ts_last_path = resume_dir / "time_series_last.pt"
        if ts_last_path.exists():
            ts_state = torch.load(ts_last_path, map_location="cpu")
            if "optimizer_state_dict" in ts_state:
                optimizer.load_state_dict(ts_state["optimizer_state_dict"])
            if "warmup_sched_state" in ts_state:
                warmup_sched.load_state_dict(ts_state["warmup_sched_state"])
            if "cosine_sched_state" in ts_state:
                cosine_sched.load_state_dict(ts_state["cosine_sched_state"])
            if "epoch_counter" in ts_state:
                _epoch_counter = ts_state["epoch_counter"]
            
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
        max_grad_norm=max_grad_norm,
        aug_scale_std=aug_scale_std,
        aug_mask_ratio=aug_mask_ratio,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_sched=warmup_sched,
        cosine_sched=cosine_sched,
        get_epoch_counter=lambda: _epoch_counter,
        use_amp=use_amp,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
        early_stopping_patience=early_stopping_patience,
        forecast_probe_epochs=forecast_probe_epochs,
        forecast_reg_lambda=forecast_reg_lambda,
        temporal_positive=temporal_positive,
        recon_lambda=recon_lambda,
        recon_mask_ratio=recon_mask_ratio,
        experiment=experiment,
        smoke=args.smoke,
    )

    experiment.end()


if __name__ == "__main__":
    main()
