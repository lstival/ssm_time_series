"""
Smoke test — CLIP-style encoder training
==========================================
Validates that the dual-encoder CLIP training pipeline is healthy:

  1. Gradients flow through both encoders and projection heads
     (per-component gradient norms logged every epoch).
  2. Contrastive loss decreases over training.
  3. Training speed (ms/batch, samples/s) is reported.

Runs 10 full epochs on all ICML datasets (ETT-small by default).
Every metric is logged to Comet ML so the run is fully reproducible.

Usage
-----
  python src/experiments/smoke_test/smoke_clip_encoder.py \
      --config src/experiments/smoke_test/smoke_config.yaml \
      --data_dir /path/to/ICML_datasets/ETT-small \
      --epochs 10 \
      --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── path setup ───────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent          # …/src
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import (
    build_projection_head,
    build_time_series_dataloaders,
    clip_contrastive_loss,
    extract_sequence,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
)


# ── gradient diagnostics ─────────────────────────────────────────────────────

def _grad_norm(module: nn.Module) -> float:
    """L2 norm of all gradients in *module* (0.0 if no grad yet)."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def _check_gradients(
    components: Dict[str, nn.Module],
) -> Dict[str, float]:
    """Return per-component gradient norms. Warn on NaN / zero."""
    norms: Dict[str, float] = {}
    for name, mod in components.items():
        norm = _grad_norm(mod)
        norms[name] = norm
        if norm == 0.0:
            print(f"  [WARN] Zero gradient in '{name}' — gradients may not be flowing!")
        for pname, p in mod.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"  [WARN] Non-finite gradient in '{name}.{pname}'!")
    return norms


# ── one training epoch ────────────────────────────────────────────────────────

def _train_epoch(
    encoder: nn.Module,
    visual: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float,
    max_grad_norm: float,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Run one epoch.

    Returns
    -------
    avg_loss, ms_per_batch, samples_per_sec, grad_norms
    """
    encoder.train(); visual.train(); proj.train(); vproj.train()

    total_loss = 0.0
    total_samples = 0
    n_batches = 0
    epoch_grad_norms: Dict[str, List[float]] = {
        "encoder": [], "visual": [], "proj": [], "vproj": []
    }

    t_start = time.perf_counter()

    for batch in loader:
        seq = extract_sequence(batch)
        if isinstance(batch, dict) and "lengths" in batch:
            seq = seq[:, : int(batch["lengths"].max().item())]
        seq = prepare_sequence(seq.to(device).float())
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))

        q = F.normalize(proj(encoder(x_q)), dim=1)
        k = F.normalize(vproj(visual(x_k)), dim=1)
        loss = clip_contrastive_loss(q, k)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Collect per-batch gradient norms before clipping
        components = {"encoder": encoder, "visual": visual,
                      "proj": proj, "vproj": vproj}
        for cname, mod in components.items():
            epoch_grad_norms[cname].append(_grad_norm(mod))

        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(visual.parameters()) +
                list(proj.parameters()) + list(vproj.parameters()),
                max_grad_norm,
            )

        optimizer.step()

        total_loss += float(loss.item())
        total_samples += x_q.shape[0]
        n_batches += 1

    elapsed = time.perf_counter() - t_start
    ms_per_batch = 1000.0 * elapsed / max(1, n_batches)
    samples_per_sec = total_samples / max(1e-9, elapsed)
    avg_loss = total_loss / max(1, n_batches)

    mean_norms = {k: (sum(v) / len(v) if v else 0.0) for k, v in epoch_grad_norms.items()}
    return avg_loss, ms_per_batch, samples_per_sec, mean_norms


# ── validation loss ───────────────────────────────────────────────────────────

@torch.no_grad()
def _val_loss(
    encoder: nn.Module,
    visual: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    device: torch.device,
    noise_std: float,
) -> float:
    encoder.eval(); visual.eval(); proj.eval(); vproj.eval()
    total, n = 0.0, 0
    for batch in loader:
        seq = extract_sequence(batch)
        if isinstance(batch, dict) and "lengths" in batch:
            seq = seq[:, : int(batch["lengths"].max().item())]
        seq = prepare_sequence(seq.to(device).float())
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))
        q = F.normalize(proj(encoder(x_q)), dim=1)
        k = F.normalize(vproj(visual(x_k)), dim=1)
        total += float(clip_contrastive_loss(q, k).item())
        n += 1
    encoder.train(); visual.train(); proj.train(); vproj.train()
    return total / max(1, n)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test: CLIP encoder training")
    p.add_argument("--config", type=Path,
                   default=script_dir / "smoke_config.yaml",
                   help="Path to experiment YAML config")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets" / "ETT-small",
                   help="Root directory with ICML CSV files for training")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of training epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_comet", action="store_true",
                   help="Disable Comet ML logging (useful for local debug)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = tu.prepare_device("auto")
    print(f"Device : {device}")
    print(f"Data   : {args.data_dir}")
    print(f"Epochs : {args.epochs}")

    # ── config ────────────────────────────────────────────────────────────────
    config = tu.load_config(args.config)
    train_cfg = config.training
    model_cfg = config.model

    # ── dataloaders ───────────────────────────────────────────────────────────
    print("\nBuilding dataloaders …")
    train_loader, val_loader = build_time_series_dataloaders(
        data_dir=args.data_dir,
        dataset_type="icml",
        batch_size=int(config.data.get("batch_size", 64)),
        val_batch_size=int(config.data.get("val_batch_size", 32)),
        num_workers=int(config.data.get("num_workers", 0)),
        pin_memory=bool(config.data.get("pin_memory", False)),
        val_ratio=float(config.data.get("val_ratio", 0.1)),
        seed=args.seed,
    )
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else "?"
    n_val   = len(val_loader.dataset)   if val_loader and hasattr(val_loader, "dataset") else "?"
    print(f"  train samples : {n_train}")
    print(f"  val   samples : {n_val}")
    print(f"  train batches : {len(train_loader)}")

    # ── models ────────────────────────────────────────────────────────────────
    print("\nBuilding models …")
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    visual  = tu.build_visual_encoder_from_config(
        model_cfg, rp_mode="correct", rp_mv_strategy="per_channel", repr_type="rp"
    ).to(device)
    proj  = build_projection_head(encoder).to(device)
    vproj = build_projection_head(visual).to(device)

    n_enc  = sum(p.numel() for p in encoder.parameters())
    n_vis  = sum(p.numel() for p in visual.parameters())
    n_proj = sum(p.numel() for p in proj.parameters())
    print(f"  temporal encoder params : {n_enc:,}")
    print(f"  visual   encoder params : {n_vis:,}")
    print(f"  projection head  params : {n_proj:,} (each)")

    # ── optimizer / scheduler ─────────────────────────────────────────────────
    all_params = (list(encoder.parameters()) + list(visual.parameters()) +
                  list(proj.parameters()) + list(vproj.parameters()))
    optimizer = torch.optim.AdamW(
        all_params,
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = tu.build_scheduler(optimizer, train_cfg, args.epochs)
    noise_std     = float(train_cfg.get("noise_std", 0.01))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    # ── Comet ML ──────────────────────────────────────────────────────────────
    experiment = None
    if not args.no_comet:
        try:
            from comet_utils import create_comet_experiment
            experiment = create_comet_experiment("smoke_clip_encoder")
            experiment.log_parameters({
                "epochs": args.epochs,
                "batch_size": config.data.get("batch_size"),
                "learning_rate": train_cfg.get("learning_rate"),
                "noise_std": noise_std,
                "model_dim": model_cfg.get("model_dim"),
                "embedding_dim": model_cfg.get("embedding_dim"),
                "depth": model_cfg.get("depth"),
                "n_train_samples": n_train,
                "seed": args.seed,
                "device": str(device),
            })
            print("\nComet ML experiment created.")
        except Exception as exc:
            print(f"\n[WARN] Comet ML init failed ({exc}). Running without logging.")

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Starting CLIP encoder training smoke test")
    print(f"{'='*60}")

    grad_zero_epochs: List[int] = []  # epochs where any component had zero grad

    wall_start = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss, ms_per_batch, sps, grad_norms = _train_epoch(
            encoder, visual, proj, vproj,
            train_loader, optimizer, device, noise_std, max_grad_norm,
        )

        val_loss: Optional[float] = None
        if val_loader is not None:
            val_loss = _val_loss(
                encoder, visual, proj, vproj, val_loader, device, noise_std
            )

        if scheduler is not None:
            if hasattr(scheduler, "step"):
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ── gradient health check ─────────────────────────────────────────────
        zero_grads = [k for k, v in grad_norms.items() if v == 0.0]
        if zero_grads:
            grad_zero_epochs.append(epoch + 1)

        # ── console report ────────────────────────────────────────────────────
        val_str = f"  val_loss={val_loss:.4f}" if val_loss is not None else ""
        print(
            f"Epoch {epoch+1:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}{val_str}  "
            f"lr={current_lr:.2e}  "
            f"speed={ms_per_batch:.1f}ms/batch  {sps:.0f}smp/s"
        )
        print(
            f"  grad_norms — encoder={grad_norms['encoder']:.3f}  "
            f"visual={grad_norms['visual']:.3f}  "
            f"proj={grad_norms['proj']:.3f}  "
            f"vproj={grad_norms['vproj']:.3f}"
        )

        # ── Comet logging ─────────────────────────────────────────────────────
        if experiment is not None:
            step = epoch + 1
            experiment.log_metric("train_loss",    train_loss,    step=step)
            experiment.log_metric("speed_ms_batch", ms_per_batch, step=step)
            experiment.log_metric("speed_smp_s",    sps,          step=step)
            experiment.log_metric("learning_rate",  current_lr,   step=step)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=step)
            for cname, norm in grad_norms.items():
                experiment.log_metric(f"grad_norm_{cname}", norm, step=step)

    total_wall = time.perf_counter() - wall_start

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Smoke test complete")
    print(f"  Total wall time  : {total_wall:.1f}s")
    print(f"  Avg ms per batch : — (see per-epoch above)")

    if grad_zero_epochs:
        print(f"  [FAIL] Zero-gradient epochs: {grad_zero_epochs}")
    else:
        print("  [PASS] Gradients flowing in all components across all epochs")

    if experiment is not None:
        experiment.log_metric("total_wall_time_s", total_wall)
        experiment.log_other("grad_zero_epochs", str(grad_zero_epochs) if grad_zero_epochs else "none")
        experiment.end()
        print("  Comet ML experiment ended.")


if __name__ == "__main__":
    main()
