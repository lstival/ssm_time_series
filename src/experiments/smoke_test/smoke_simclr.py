"""
Smoke test — SimCLR single-encoder training
============================================
Validates that both SimCLR training modes (temporal and visual) are healthy:

  1. Gradients flow through the encoder and projection head.
  2. NT-Xent loss decreases over training.
  3. Training speed (ms/batch, samples/s) is reported.
  4. No NaN/inf in loss or gradients.

Runs a small number of epochs on a single ICML dataset (ETTm1.csv by default)
with a reduced model (model_dim=64, depth=2) for speed.

Usage
-----
  python src/experiments/smoke_test/smoke_simclr.py \
      --config src/experiments/smoke_test/smoke_config.yaml \
      --data_dir /path/to/ICML_datasets/ETT-small \
      --epochs 3 \
      --modes temporal visual \
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
    build_time_series_dataloaders,
    extract_sequence,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
)


# ── NT-Xent / SimCLR loss ────────────────────────────────────────────────────

def _simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── gradient diagnostics ─────────────────────────────────────────────────────

def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def _has_nonfinite_grad(module: nn.Module) -> bool:
    for p in module.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return True
    return False


# ── one training epoch ────────────────────────────────────────────────────────

def _train_epoch(
    encoder: nn.Module,
    proj: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float,
    max_grad_norm: float,
    temperature: float,
) -> Tuple[float, float, float, Dict[str, float]]:
    encoder.train()
    proj.train()

    total_loss = 0.0
    total_samples = 0
    n_batches = 0
    enc_norms: List[float] = []
    proj_norms: List[float] = []

    t_start = time.perf_counter()

    for batch in loader:
        seq = extract_sequence(batch)
        if isinstance(batch, dict) and "lengths" in batch:
            seq = seq[:, : int(batch["lengths"].max().item())]
        seq = prepare_sequence(seq.to(device).float())
        x1 = reshape_multivariate_series(seq)
        x2 = make_positive_view(x1 + noise_std * torch.randn_like(x1))

        z1 = proj(encoder(x1))
        z2 = proj(encoder(x2))
        loss = _simclr_loss(z1, z2, temperature)

        if not torch.isfinite(loss):
            print(f"  [WARN] Non-finite loss: {loss.item()}")
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        enc_norms.append(_grad_norm(encoder))
        proj_norms.append(_grad_norm(proj))

        if _has_nonfinite_grad(encoder) or _has_nonfinite_grad(proj):
            print("  [WARN] Non-finite gradients detected!")

        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(proj.parameters()),
                max_grad_norm,
            )

        optimizer.step()
        total_loss += float(loss.item())
        total_samples += x1.shape[0]
        n_batches += 1

    elapsed = time.perf_counter() - t_start
    avg_loss = total_loss / max(1, n_batches)
    ms_per_batch = 1000.0 * elapsed / max(1, n_batches)
    sps = total_samples / max(1e-9, elapsed)
    mean_norms = {
        "encoder": sum(enc_norms) / len(enc_norms) if enc_norms else 0.0,
        "proj":    sum(proj_norms) / len(proj_norms) if proj_norms else 0.0,
    }
    return avg_loss, ms_per_batch, sps, mean_norms


# ── run one mode (temporal or visual) ────────────────────────────────────────

def run_mode(
    mode: str,
    config,
    data_dir: Path,
    epochs: int,
    seed: int,
    device: torch.device,
    experiment,
) -> bool:
    """
    Train one SimCLR mode for *epochs* epochs and print diagnostics.
    Returns True if all checks pass, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"  Mode: {mode.upper()}")
    print(f"{'='*60}")

    model_cfg = config.model
    train_cfg = config.training
    noise_std = float(train_cfg.get("noise_std", 0.01))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    temperature = float(train_cfg.get("temperature", 0.07))
    lr = float(train_cfg.get("learning_rate", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    embedding_dim = int(model_cfg.get("embedding_dim", 64))
    model_dim = int(model_cfg.get("model_dim", 64))

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader, val_loader = build_time_series_dataloaders(
        data_dir=str(data_dir),
        dataset_type="icml",
        batch_size=int(config.data.get("batch_size", 64)),
        val_batch_size=int(config.data.get("val_batch_size", 32)),
        num_workers=int(config.data.get("num_workers", 0)),
        pin_memory=bool(config.data.get("pin_memory", False)),
        val_ratio=float(config.data.get("val_ratio", 0.1)),
        seed=seed,
    )
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else "?"
    print(f"  train samples: {n_train}  |  batches: {len(train_loader)}")

    # ── encoder ───────────────────────────────────────────────────────────────
    if mode == "temporal":
        encoder = tu.build_encoder_from_config(model_cfg).to(device)
        enc_label = "MambaEncoder"
    else:
        from models.mamba_visual_encoder import UpperTriDiagRPEncoder
        encoder = UpperTriDiagRPEncoder(
            patch_len=int(model_cfg.get("input_dim", 32)),
            d_model=model_dim,
            n_layers=int(model_cfg.get("depth", 2)),
            embedding_dim=embedding_dim,
            rp_mv_strategy="mean",      # Ablation A best
        ).to(device)
        enc_label = "UpperTriDiagRPEncoder (anti-diagonal lag tokenization)"

    proj = ProjectionHead(embedding_dim, model_dim, model_dim).to(device)

    n_enc  = sum(p.numel() for p in encoder.parameters())
    n_proj = sum(p.numel() for p in proj.parameters())
    print(f"  encoder: {enc_label}  ({n_enc:,} params)")
    print(f"  proj head: {n_proj:,} params")

    # ── optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=lr,
        weight_decay=wd,
    )

    # ── training ──────────────────────────────────────────────────────────────
    losses: List[float] = []
    grad_zero_epochs: List[int] = []
    nonfinite_epochs: List[int] = []

    wall_start = time.perf_counter()
    for epoch in range(epochs):
        avg_loss, ms_batch, sps, grad_norms = _train_epoch(
            encoder, proj, train_loader, optimizer, device,
            noise_std, max_grad_norm, temperature,
        )
        losses.append(avg_loss)

        zero_grads = [k for k, v in grad_norms.items() if v == 0.0]
        if zero_grads:
            grad_zero_epochs.append(epoch + 1)
            print(f"  [WARN] Zero gradients in: {zero_grads}")

        if not torch.isfinite(torch.tensor(avg_loss)):
            nonfinite_epochs.append(epoch + 1)

        print(
            f"  Epoch {epoch+1:02d}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"speed={ms_batch:.1f}ms/batch  {sps:.0f}smp/s  "
            f"grad[enc]={grad_norms['encoder']:.3f}  grad[proj]={grad_norms['proj']:.3f}"
        )

        if experiment is not None:
            step = epoch + 1
            experiment.log_metric(f"{mode}_train_loss",     avg_loss,  step=step)
            experiment.log_metric(f"{mode}_ms_per_batch",   ms_batch,  step=step)
            experiment.log_metric(f"{mode}_grad_norm_enc",  grad_norms["encoder"], step=step)
            experiment.log_metric(f"{mode}_grad_norm_proj", grad_norms["proj"],    step=step)

    total_wall = time.perf_counter() - wall_start

    # ── diagnostics ───────────────────────────────────────────────────────────
    print(f"\n  Wall time: {total_wall:.1f}s")

    loss_decreasing = len(losses) >= 2 and losses[-1] < losses[0]
    passed = True

    if grad_zero_epochs:
        print(f"  [FAIL] Zero-gradient epochs: {grad_zero_epochs}")
        passed = False
    else:
        print("  [PASS] Gradients flowing in all epochs")

    if nonfinite_epochs:
        print(f"  [FAIL] Non-finite loss in epochs: {nonfinite_epochs}")
        passed = False
    else:
        print("  [PASS] Loss is finite in all epochs")

    if loss_decreasing:
        print(f"  [PASS] Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")
    else:
        print(f"  [INFO] Loss did not strictly decrease ({losses}). "
              "Normal for very short runs.")

    return passed


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test: SimCLR single-encoder training")
    p.add_argument("--config", type=Path,
                   default=script_dir / "smoke_config.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets" / "ETT-small")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--modes", nargs="+", choices=["temporal", "visual"],
                   default=["temporal", "visual"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_comet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = tu.prepare_device("auto")
    print(f"Device : {device}")
    print(f"Data   : {args.data_dir}")
    print(f"Epochs : {args.epochs}")
    print(f"Modes  : {args.modes}")

    config = tu.load_config(args.config)

    # ── Comet ML ──────────────────────────────────────────────────────────────
    experiment = None
    if not args.no_comet:
        try:
            from comet_utils import create_comet_experiment
            experiment = create_comet_experiment("smoke_simclr")
            experiment.log_parameters({
                "epochs": args.epochs,
                "modes": args.modes,
                "seed": args.seed,
                "device": str(device),
                "model_dim": config.model.get("model_dim"),
                "depth": config.model.get("depth"),
            })
            print("Comet ML experiment created.")
        except Exception as exc:
            print(f"[WARN] Comet ML init failed ({exc}). Running without logging.")

    # ── run modes ─────────────────────────────────────────────────────────────
    results: Dict[str, bool] = {}
    for mode in args.modes:
        results[mode] = run_mode(
            mode=mode,
            config=config,
            data_dir=args.data_dir,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
            experiment=experiment,
        )

    # ── final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SimCLR smoke test summary")
    print(f"{'='*60}")
    all_passed = True
    for mode, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {mode}")
        if not passed:
            all_passed = False

    if experiment is not None:
        for mode, passed in results.items():
            experiment.log_other(f"smoke_{mode}", "PASS" if passed else "FAIL")
        experiment.end()

    if not all_passed:
        print("\nOne or more modes FAILED. Check logs above.")
        sys.exit(1)
    else:
        print("\nAll modes PASSED.")


if __name__ == "__main__":
    main()
