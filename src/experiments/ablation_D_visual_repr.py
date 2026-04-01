"""
Ablation D — Visual Representations on Multiple Datasets
==========================================================
Extends Table 5 (currently ETTh1-only) to five datasets and adds a
computational cost column.

  Representations: RP (baseline), GASF, MTF, STFT
  Datasets       : ETTh1, ETTm1, Weather, Traffic, Solar
  Metric         : MSE / MAE (linear probe, H=96) + ms/batch

RP must be Pareto-dominant (≤ MSE, lower cost) on ≥ 4/5 datasets.

Usage
-----
  python src/experiments/ablation_D_visual_repr.py \
      --config src/configs/lotsa_clip.yaml \
      --train_epochs 20 --probe_epochs 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as torchF
import torch.nn as nn

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
from time_series_loader import TimeSeriesDataModule

REPR_TYPES: List[str] = ["rp", "gasf", "mtf", "stft"]
PROBE_DATASETS: List[str] = ["ETTh1.csv", "ETTm1.csv", "weather.csv", "traffic.csv", "solar_AL.txt"]
HORIZON: int = 96  # Table 5 uses H=96 as primary metric


def _clip_train_epoch(
    encoder: nn.Module,
    visual: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.01,
) -> float:
    encoder.train(); visual.train(); proj.train(); vproj.train()
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

        q = torchF.normalize(proj(encoder(x_q)), dim=1)
        k = torchF.normalize(vproj(visual(x_k)), dim=1)
        loss = clip_contrastive_loss(q, k)

        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(visual.parameters())
            + list(proj.parameters()) + list(vproj.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        total += float(loss.item()); n += 1

    return total / max(1, n)


def _measure_ms_per_batch(visual: nn.Module, loader, device: torch.device) -> float:
    """Measure average forward-pass time (ms) for the visual encoder."""
    visual.eval()
    times = []
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        if isinstance(batch, dict) and "target" in batch:
            seq = batch["target"].to(device).float()
        elif isinstance(batch, (tuple, list)):
            seq = batch[0].to(device).float()
        else:
            seq = batch.to(device).float()
        seq = prepare_sequence(seq)
        x = reshape_multivariate_series(seq)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = visual(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(sum(times) / max(1, len(times)))


def _probe_evaluate(
    encoder: nn.Module,
    visual: nn.Module,
    data_dir: Path,
    dataset_csv: str,
    horizon: int,
    device: torch.device,
    probe_epochs: int = 30,
) -> Dict[str, float]:
    module = TimeSeriesDataModule(
        dataset_name=dataset_csv,
        data_dir=str(data_dir),
        batch_size=64, val_batch_size=64,
        num_workers=0, pin_memory=False,
        normalize=True, train=True, val=False, test=True,
    )
    module.setup()
    if not module.train_loaders:
        return {"mse": float("nan"), "mae": float("nan")}
    train_loader = module.train_loaders[0]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    encoder.eval(); visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    def _embed(batch):
        x = batch[0].to(device).float()
        x = reshape_multivariate_series(prepare_sequence(x))
        with torch.no_grad():
            ze = encoder(x)
            zv = visual(x)
            ze = torch.nan_to_num(ze, nan=0.0, posinf=0.0, neginf=0.0)
            zv = torch.nan_to_num(zv, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.cat([ze, zv], dim=1)

    def _collect(loader):
        zs, ys = [], []
        for b in loader:
            try:
                z = _embed(b)
                # Validate embedding
                if not torch.isfinite(z).all():
                    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                zs.append(z)

                # Extract target (second element of batch tuple)
                # Handle both (x, y, ...) and (x, y) formats from dataloader
                y = b[1].to(device).float() if len(b) > 1 else None
                if y is None:
                    continue

                # Ensure target has correct shape: (batch, seq_len, features) or (batch, seq_len)
                if y.ndim > 2:
                    y = y[:, :, 0]  # Take first feature if multi-feature target

                # Validate target
                if not torch.isfinite(y).all():
                    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

                ys.append(y)
            except Exception as e:
                print(f"Warning: Skipping batch due to error: {e}", file=sys.stderr)
                continue

        if not ys or not zs:
            # Return empty tensors if no valid data
            empty_z = torch.empty((0, 256), device=device)  # Embedding dim
            empty_y = torch.empty((0, horizon), device=device)
            return empty_z, empty_y

        return torch.cat(zs), torch.cat(ys)

    Z_tr, Y_tr = _collect(train_loader)
    probe = nn.Linear(Z_tr.shape[1], horizon).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(probe_epochs):
        probe.train()
        perm = torch.randperm(Z_tr.shape[0], device=device)
        for i in range(0, Z_tr.shape[0], 64):
            idx = perm[i:i+64]
            z_b = Z_tr[idx]; y_b = Y_tr[idx]

            # Validate and reshape targets
            if y_b.ndim == 3: y_b = y_b[:, :horizon, 0]
            elif y_b.ndim == 2: y_b = y_b[:, :horizon]
            else:
                # Skip batches with unexpected shapes
                continue

            # Ensure batch has enough temporal points
            if y_b.shape[1] < horizon:
                continue

            # Clip values to prevent NaN in gradients
            z_b = torch.clamp(z_b, -1e2, 1e2)
            y_b = torch.clamp(y_b, -1e2, 1e2)

            # Compute prediction
            pred = probe(z_b)

            # Validate prediction
            if not torch.isfinite(pred).all() or not torch.isfinite(y_b).all():
                opt.zero_grad(set_to_none=True)
                continue

            # Compute loss with numerical stability
            loss = torchF.mse_loss(pred, y_b)

            # Skip if loss is NaN/inf
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Clip gradients to prevent overflow
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=10.0)

            opt.step()

    result = {"mse": float("nan"), "mae": float("nan")}
    if test_loader:
        Z_test, Y_test = _collect(test_loader)
        probe.eval()
        with torch.no_grad():
            pred = probe(Z_test)
            if Y_test.ndim == 3: Y_test = Y_test[:, :horizon, 0]
            elif Y_test.ndim == 2: Y_test = Y_test[:, :horizon]
            if Y_test.shape[1] >= horizon:
                result["mse"] = torchF.mse_loss(pred, Y_test).item()
                result["mae"] = (pred - Y_test).abs().mean().item()

    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(True)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_D")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--pretrain_data_dir", type=Path,
                   default=src_dir.parent / "data",
                   help="Data dir for CLIP pre-training (override for smoke tests)")
    p.add_argument("--repr_types", nargs="+", default=REPR_TYPES,
                   choices=REPR_TYPES)
    p.add_argument("--probe_datasets", nargs="+", default=PROBE_DATASETS,
                   help="CSV filenames to use for linear-probe evaluation")
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

    for repr_type in args.repr_types:
        print(f"\n{'='*60}\nrepr_type: {repr_type}\n{'='*60}")
        tu.set_seed(args.seed)

        encoder = tu.build_encoder_from_config(config.model).to(device)
        visual = tu.build_visual_encoder_from_config(
            config.model, repr_type=repr_type
        ).to(device)
        proj = build_projection_head(encoder).to(device)
        vproj = build_projection_head(visual).to(device)

        params = (list(encoder.parameters()) + list(visual.parameters())
                  + list(proj.parameters()) + list(vproj.parameters()))
        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

        t0 = time.time()
        for ep in range(args.train_epochs):
            loss = _clip_train_epoch(
                encoder, visual, proj, vproj, train_loader, optimizer, device,
                noise_std=float(config.training.get("noise_std", 0.01)),
            )
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  epoch {ep+1}/{args.train_epochs}  loss={loss:.4f}")
        elapsed = time.time() - t0

        ms_per_batch = _measure_ms_per_batch(visual, train_loader, device)
        print(f"  Pre-training done — {elapsed:.0f}s | visual fwd {ms_per_batch:.1f} ms/batch")

        for ds in args.probe_datasets:
            m = _probe_evaluate(
                encoder, visual,
                data_dir=args.data_dir, dataset_csv=ds,
                horizon=HORIZON, device=device,
                probe_epochs=args.probe_epochs,
            )
            rows.append({
                "repr_type": repr_type, "dataset": ds.replace(".csv", ""),
                "horizon": HORIZON, "mse": f"{m['mse']:.4f}", "mae": f"{m['mae']:.4f}",
                "ms_per_batch": f"{ms_per_batch:.1f}",
            })
            print(f"  {ds}  H={HORIZON}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    out_csv = args.results_dir / "ablation_D_results.csv"
    fieldnames = ["repr_type", "dataset", "horizon", "mse", "mae", "ms_per_batch"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
