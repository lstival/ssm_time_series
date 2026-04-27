"""
Ablation J — SSM Hyperparameters (state_dim, conv_kernel)
===========================================================
Validates the default SSM choices used across tiers:

  - state_dim (d_SSM) = 16
  - conv_kernel (k) = 4

by sweeping smaller and larger values on the smallest model available
(`lotsa_nano`) and comparing downstream probe metrics.

Protocol
--------
  1. For each (state_dim, conv_kernel) pair, train the multimodal encoder
     for `--train_epochs` epochs.
  2. Freeze encoders and run linear probes on ETTm1 / Weather / Traffic
     for H in {96, 192, 336, 720}.
  3. Save a CSV with MSE/MAE and throughput proxy (ms/batch).

Usage
-----
  python src/experiments/ablation_J_ssm_hparams.py \
      --config src/configs/lotsa_nano.yaml \
      --train_epochs 20 \
      --probe_epochs 20 \
      --state_dims 8 16 32 \
      --conv_kernels 2 4 8
"""

from __future__ import annotations

import argparse
import csv
import time
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as torchF

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from time_series_loader import TimeSeriesDataModule
from util import (
    build_projection_head,
    build_time_series_dataloaders,
    clip_contrastive_loss,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
)

DEFAULT_STATE_DIMS: List[int] = [8, 16, 32]
DEFAULT_CONV_KERNELS: List[int] = [2, 4, 8]
PROBE_DATASETS: List[str] = ["ETTm1.csv", "weather.csv", "traffic.csv"]
HORIZONS: List[int] = [96, 192, 336, 720]


def _clip_train_epoch(encoder, visual, proj, vproj, loader, optimizer, device, noise_std=0.01):
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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(1, n)


def _probe_evaluate(encoder, visual, data_dir, dataset_csv, horizons, device, probe_epochs=20):
    max_horizon = max(horizons)
    module = TimeSeriesDataModule(
        dataset_name=dataset_csv,
        data_dir=str(data_dir),
        batch_size=64,
        val_batch_size=64,
        num_workers=0,
        pin_memory=False,
        normalize=True,
        train=True,
        val=False,
        test=True,
        sample_size=(96, 0, max_horizon),
    )
    module.setup()
    if not module.train_loaders:
        return {}

    train_loader = module.train_loaders[0]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    encoder.eval(); visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    def _embed(batch):
        x = batch[0].to(device).float()
        x = reshape_multivariate_series(prepare_sequence(x))
        with torch.no_grad():
            return torch.cat([encoder(x), visual(x)], dim=1)

    def _collect(loader):
        zs, ys = [], []
        for b in loader:
            zs.append(_embed(b))
            y = b[1].to(device).float() if len(b) > 1 else b[0].to(device).float()
            ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    Z_tr, Y_tr = _collect(train_loader)
    results: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        probe = nn.Linear(Z_tr.shape[1], H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(probe_epochs):
            probe.train()
            perm = torch.randperm(Z_tr.shape[0], device=device)
            for i in range(0, Z_tr.shape[0], 64):
                idx = perm[i:i + 64]
                z_b = Z_tr[idx]
                y_b = Y_tr[idx]
                if y_b.ndim == 3:
                    y_b = y_b[:, :H, 0]
                elif y_b.ndim == 2:
                    y_b = y_b[:, :H]
                if y_b.shape[1] < H:
                    continue
                loss = torchF.mse_loss(probe(z_b), y_b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if test_loader:
            Z_te, Y_te = _collect(test_loader)
            probe.eval()
            with torch.no_grad():
                pred = probe(Z_te)
                if Y_te.ndim == 3:
                    Y_te = Y_te[:, :H, 0]
                elif Y_te.ndim == 2:
                    Y_te = Y_te[:, :H]
                if Y_te.shape[1] < H:
                    results[H] = {"mse": float("nan"), "mae": float("nan")}
                    continue
                results[H] = {
                    "mse": torchF.mse_loss(pred, Y_te).item(),
                    "mae": (pred - Y_te).abs().mean().item(),
                }
        else:
            results[H] = {"mse": float("nan"), "mae": float("nan")}

    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(True)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation J: SSM state_dim and conv_kernel sweep")
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "lotsa_nano.yaml")
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--probe_epochs", type=int, default=20)
    p.add_argument("--results_dir", type=Path, default=src_dir.parent / "results" / "ablation_J")
    p.add_argument("--data_dir", type=Path, default=src_dir.parent / "ICML_datasets")
    p.add_argument(
        "--pretrain_data_dir",
        type=Path,
        default=src_dir.parent / "data",
        help="Data dir for pre-training (override for smoke tests)",
    )
    p.add_argument("--state_dims", nargs="+", type=int, default=DEFAULT_STATE_DIMS)
    p.add_argument("--conv_kernels", nargs="+", type=int, default=DEFAULT_CONV_KERNELS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    model_cfg_base = dict(config.model)
    if "rp_encoder" in model_cfg_base:
        # Ensure both temporal and visual branches use state_dim/conv_kernel from config.
        print("[WARN] Detected rp_encoder in config; removing it for Ablation J sweep.")
        model_cfg_base.pop("rp_encoder", None)

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

    rows: List[Dict[str, str]] = []

    for state_dim in args.state_dims:
        if state_dim <= 0:
            continue
        for conv_kernel in args.conv_kernels:
            if conv_kernel <= 0:
                continue

            print(f"\n{'=' * 66}")
            print(f"state_dim={state_dim}, conv_kernel={conv_kernel}")
            print(f"{'=' * 66}")
            tu.set_seed(args.seed)

            model_cfg = dict(model_cfg_base)
            model_cfg["state_dim"] = int(state_dim)
            model_cfg["conv_kernel"] = int(conv_kernel)

            encoder = tu.build_encoder_from_config(model_cfg).to(device)
            visual = tu.build_visual_encoder_from_config(model_cfg).to(device)
            proj = build_projection_head(encoder).to(device)
            vproj = build_projection_head(visual).to(device)

            params = (
                list(encoder.parameters())
                + list(visual.parameters())
                + list(proj.parameters())
                + list(vproj.parameters())
            )
            optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

            t0 = time.time()
            for ep in range(args.train_epochs):
                loss = _clip_train_epoch(
                    encoder,
                    visual,
                    proj,
                    vproj,
                    train_loader,
                    optimizer,
                    device,
                    noise_std=float(config.training.get("noise_std", 0.01)),
                )
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(f"  epoch {ep + 1}/{args.train_epochs}  loss={loss:.4f}")
            elapsed = time.time() - t0
            ms_per_batch = 1000.0 * elapsed / (args.train_epochs * max(1, len(train_loader)))
            print(f"  Done - {elapsed:.0f}s ({ms_per_batch:.1f} ms/batch)")

            for ds in PROBE_DATASETS:
                metrics = _probe_evaluate(
                    encoder=encoder,
                    visual=visual,
                    data_dir=args.data_dir,
                    dataset_csv=ds,
                    horizons=HORIZONS,
                    device=device,
                    probe_epochs=args.probe_epochs,
                )
                for horizon, metric in metrics.items():
                    rows.append(
                        {
                            "state_dim": str(state_dim),
                            "conv_kernel": str(conv_kernel),
                            "dataset": ds.replace(".csv", ""),
                            "horizon": str(horizon),
                            "mse": f"{metric['mse']:.4f}",
                            "mae": f"{metric['mae']:.4f}",
                            "ms_per_batch": f"{ms_per_batch:.1f}",
                        }
                    )
                    print(
                        f"  {ds:12s} H={horizon:3d} "
                        f"MSE={metric['mse']:.4f} MAE={metric['mae']:.4f}"
                    )

    out_csv = args.results_dir / "ablation_J_results.csv"
    fieldnames = ["state_dim", "conv_kernel", "dataset", "horizon", "mse", "mae", "ms_per_batch"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")

    # Report best setting by average MSE across all datasets/horizons.
    scores: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        try:
            key = f"d{row['state_dim']}_k{row['conv_kernel']}"
            scores[key].append(float(row["mse"]))
        except ValueError:
            continue
    if scores:
        best_key = min(scores, key=lambda k: sum(scores[k]) / max(1, len(scores[k])))
        best_avg = sum(scores[best_key]) / max(1, len(scores[best_key]))
        print(f"Recommended setting (lowest avg MSE): {best_key} with avg MSE={best_avg:.4f}")


if __name__ == "__main__":
    main()
