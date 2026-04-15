"""
Ablation C — Contrastive Alignment Strategy
=============================================
Compares four alignment strategies to justify CLIP-style training:

  Variant              | Description
  ---------------------|-----------------------------------------------------
  clip_symm  (base)    | Symmetric NT-Xent CLIP loss (current)
  cosine_mse           | MSE between L2-normalised temporal and visual embeddings
  concat_supervised    | Concatenated embeddings + supervised MLP (no SSL step)
  unimodal_temporal    | Only temporal encoder, no visual branch (lower bound)

Key question: clip_symm must outperform cosine_mse to justify the choice.
concat_supervised is the dangerous baseline — if it wins, the SSL contribution
is questionable.

Protocol
--------
  1. Train each variant on LOTSA for `--train_epochs` epochs.
  2. Linear-probe evaluation on ETTm1, Weather, Exchange for H ∈ {96, 192, 336, 720}.
  3. Results written to results/ablation_C_alignment.csv.

Usage
-----
  python src/experiments/ablation_C_alignment.py \
      --config src/configs/lotsa_clip.yaml \
      --train_epochs 20 --probe_epochs 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torchF

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import (
    build_time_series_dataloaders,
    clip_contrastive_loss,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
    build_projection_head,
)
from models.mamba_encoder import MambaEncoder
from models.mamba_visual_encoder import MambaVisualEncoder
from time_series_loader import TimeSeriesDataModule

VARIANTS: List[str] = ["clip_symm", "cosine_mse", "concat_supervised", "unimodal_temporal"]
PROBE_DATASETS: List[str] = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv", "exchange_rate.csv",
]
HORIZONS: List[int] = [96, 192, 336, 720]
SEQ_LEN: int = 336  # match final models (simclr-best / clip-best-nano)


# ── loss functions ────────────────────────────────────────────────────────────

def cosine_mse_loss(z_q: torch.Tensor, z_k: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalised embeddings (minimises angle between them)."""
    q = torchF.normalize(z_q, dim=1)
    k = torchF.normalize(z_k, dim=1)
    return torchF.mse_loss(q, k)


# ── concat-supervised model ───────────────────────────────────────────────────

class ConcatSupervisedHead(nn.Module):
    """Linear head on top of [z_temporal ‖ z_visual] for supervised forecasting."""

    def __init__(self, feat_dim: int, horizon: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# ── training helpers ──────────────────────────────────────────────────────────

def _train_ssl_epoch(
    encoder: nn.Module,
    visual: Optional[nn.Module],
    proj: nn.Module,
    vproj: Optional[nn.Module],
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.01,
    variant: str = "clip_symm",
) -> float:
    encoder.train(); proj.train()
    if visual is not None: visual.train()
    if vproj is not None:  vproj.train()

    total, n = 0.0, 0
    for batch in loader:
        if isinstance(batch, dict) and "target" in batch:
            seq = batch["target"].to(device).float()
            if "lengths" in batch:
                seq = seq[:, : int(batch["lengths"].max().item())]
        elif isinstance(batch, (tuple, list)):
            seq = batch[0].to(device).float()
        else:
            seq = batch.to(device).float()

        seq = prepare_sequence(seq)
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))

        z_q = encoder(x_q)
        z_k = visual(x_k) if visual is not None else encoder(x_k)

        if variant == "clip_symm":
            q = torchF.normalize(proj(z_q), dim=1)
            k = torchF.normalize(vproj(z_k), dim=1)
            loss = clip_contrastive_loss(q, k)

        elif variant == "cosine_mse":
            q = proj(z_q)
            k = vproj(z_k)
            loss = cosine_mse_loss(q, k)

        elif variant == "unimodal_temporal":
            # Two augmented views through same temporal encoder
            q = torchF.normalize(proj(z_q), dim=1)
            k = torchF.normalize(proj(encoder(x_k)), dim=1)
            loss = clip_contrastive_loss(q, k)

        else:
            raise ValueError(f"Unexpected variant in SSL loop: {variant}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()); n += 1

    return total / max(1, n)


def _train_supervised_epoch(
    encoder: nn.Module,
    visual: nn.Module,
    head: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    horizon: int,
) -> float:
    """One epoch of supervised training for concat_supervised variant."""
    encoder.train(); visual.train(); head.train()
    total, n = 0.0, 0
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            continue
        x = batch[0].to(device).float()
        y = batch[1].to(device).float()

        x = prepare_sequence(x)
        xr = reshape_multivariate_series(x)
        with torch.no_grad():
            ze = encoder(xr)
            zv = visual(xr)
        z = torch.cat([ze, zv], dim=1)
        pred = head(z)

        if y.ndim == 3: y = y[:, :horizon, 0]
        elif y.ndim == 2: y = y[:, :horizon]
        if y.shape[1] < horizon: continue

        loss = torchF.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()); n += 1

    return total / max(1, n)


# ── linear probe ──────────────────────────────────────────────────────────────

def _probe_evaluate(
    encoder: nn.Module,
    visual: Optional[nn.Module],
    data_dir: Path,
    dataset_csv: str,
    horizons: List[int],
    device: torch.device,
    probe_epochs: int = 30,
    variant: str = "clip_symm",
) -> Dict[int, Dict[str, float]]:
    max_horizon = max(horizons)
    module = TimeSeriesDataModule(
        dataset_name=dataset_csv,
        data_dir=str(data_dir),
        batch_size=64,
        val_batch_size=64,
        num_workers=0,
        pin_memory=False,
        normalize=True,
        train=True, val=False, test=True,
        sample_size=(SEQ_LEN, 0, max_horizon),
        scaler_type="standard",
    )
    module.setup()
    if not module.train_loaders:
        return {}
    train_loader = module.train_loaders[0]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    encoder.eval()
    if visual is not None: visual.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    if visual is not None:
        for p in visual.parameters(): p.requires_grad_(False)

    def _embed(batch):
        x = batch[0].to(device).float()
        x = reshape_multivariate_series(prepare_sequence(x))
        with torch.no_grad():
            ze = encoder(x)
            zv = visual(x) if visual is not None else ze
        return torch.cat([ze, zv], dim=1)

    def _collect(loader):
        zs, ys = [], []
        for batch in loader:
            zs.append(_embed(batch))
            y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
            ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    Z_tr, Y_tr = _collect(train_loader)
    feat_dim = Z_tr.shape[1]
    results: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        probe = nn.Linear(feat_dim, H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(probe_epochs):
            probe.train()
            perm = torch.randperm(Z_tr.shape[0], device=device)
            for i in range(0, Z_tr.shape[0], 64):
                idx = perm[i:i+64]
                z_b = Z_tr[idx]; y_b = Y_tr[idx]
                if y_b.ndim == 3: y_b = y_b[:, :H, 0]
                elif y_b.ndim == 2: y_b = y_b[:, :H]
                if y_b.shape[1] < H: continue
                loss = torchF.mse_loss(probe(z_b), y_b)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if test_loader:
            Z_test, Y_test = _collect(test_loader)
            probe.eval()
            with torch.no_grad():
                pred = probe(Z_test)
                if Y_test.ndim == 3: Y_test = Y_test[:, :H, 0]
                elif Y_test.ndim == 2: Y_test = Y_test[:, :H]
                if Y_test.shape[1] < H:
                    results[H] = {"mse": float("nan"), "mae": float("nan")}
                    continue
                results[H] = {
                    "mse": torchF.mse_loss(pred, Y_test).item(),
                    "mae": (pred - Y_test).abs().mean().item(),
                }
        else:
            results[H] = {"mse": float("nan"), "mae": float("nan")}

    for p in encoder.parameters(): p.requires_grad_(True)
    if visual is not None:
        for p in visual.parameters(): p.requires_grad_(True)
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip_nano.yaml")
    p.add_argument("--train_epochs", type=int, default=100)
    p.add_argument("--probe_epochs", type=int, default=20)
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_C")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--pretrain_data_dir", type=Path,
                   default=src_dir.parent / "data",
                   help="Data dir for CLIP pre-training (override for smoke tests)")
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _ = build_time_series_dataloaders(
        data_dir=str(args.pretrain_data_dir),
        dataset_name=config.data.get("dataset_name", ""),
        dataset_type=config.data.get("dataset_type", "lotsa"),
        batch_size=int(config.data.get("batch_size", 256)),
        val_batch_size=int(config.data.get("val_batch_size", 128)),
        num_workers=int(config.data.get("num_workers", 4)),
        pin_memory=bool(config.data.get("pin_memory", True)),
        val_ratio=float(config.data.get("val_ratio", 0.1)),
        cronos_kwargs=dict(config.data.get("cronos_kwargs", {})),
        seed=args.seed,
    )

    rows: List[Dict] = []

    for variant in args.variants:
        print(f"\n{'='*60}\nVariant: {variant}\n{'='*60}")
        tu.set_seed(args.seed)

        encoder = tu.build_encoder_from_config(config.model).to(device)
        proj = build_projection_head(encoder).to(device)
        params = list(encoder.parameters()) + list(proj.parameters())

        if variant in ("clip_symm", "cosine_mse"):
            visual = tu.build_visual_encoder_from_config(config.model).to(device)
            vproj = build_projection_head(visual).to(device)
            params += list(visual.parameters()) + list(vproj.parameters())
        elif variant == "concat_supervised":
            visual = tu.build_visual_encoder_from_config(config.model).to(device)
            vproj = None
            # encoders are frozen during probe; no additional params for SSL
        else:  # unimodal_temporal
            visual = None
            vproj = None

        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

        t0 = time.time()
        if variant != "concat_supervised":
            for ep in range(args.train_epochs):
                loss = _train_ssl_epoch(
                    encoder, visual, proj, vproj, train_loader,
                    optimizer, device,
                    noise_std=float(config.training.get("noise_std", 0.01)),
                    variant=variant,
                )
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(f"  epoch {ep+1}/{args.train_epochs}  loss={loss:.4f}")
        else:
            print("  concat_supervised: encoders evaluated in frozen mode only")
        elapsed = time.time() - t0
        print(f"  Done — {elapsed:.0f}s")

        for ds in PROBE_DATASETS:
            metrics = _probe_evaluate(
                encoder, visual,
                data_dir=args.data_dir, dataset_csv=ds,
                horizons=HORIZONS, device=device,
                probe_epochs=args.probe_epochs,
                variant=variant,
            )
            for H, m in metrics.items():
                rows.append({
                    "variant": variant, "dataset": ds.replace(".csv", ""),
                    "horizon": H, "mse": f"{m['mse']:.4f}", "mae": f"{m['mae']:.4f}",
                })
                print(f"  {ds}  H={H:3d}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    out_csv = args.results_dir / "ablation_C_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "dataset", "horizon", "mse", "mae"])
        writer.writeheader(); writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
