"""
Ablation H — Visual Encoder Architecture for Recurrence Plots
============================================================
Determines which architecture processes the 2D structure of Recurrence Plots 
best for small patch sizes l ∈ {16, 32, 64}.

Variants:
  1. CNN Baseline (local filters)
  2. Flatten + Mamba 1D (baseline destruction of 2D structure)
  3. RP-SS2D (2 scans: horizontal/vertical) - Proposed
  4. SS2D Full (4 scans: horizontal/vertical forward/backward)

Protocol:
  1. For each (variant, l), pre-train CM-Mamba (CLIP) on LOTSA.
  2. Linear-probe on ETTm1, Weather, Traffic for H=96.
  3. Measure: MSE, MAE, ms/batch, Parameters, Peak GPU Memory.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from models.mamba_visual_encoder import VisualEncoderFactory, tokenize_sequence
from time_series_loader import TimeSeriesDataModule

ENCODER_VARIANTS = ["cnn", "flatten_mamba", "rp_ss2d_2", "ss2d_4", "upper_tri_diag"]
PATCH_LENGTHS = [16, 32, 64]
PROBE_DATASETS = ["ETTm1.csv", "weather.csv", "traffic.csv"]
HORIZON = 96


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
        
        # Compute RP for visual branch
        with torch.no_grad():
            tokens = tokenize_sequence(x_k, token_size=visual.patch_len)
            B, W, L, F = tokens.shape
            # (B*W, F, L)
            tokens_for_rp = tokens.permute(0, 1, 3, 2).reshape(B * W, F, L)
            # Use GPU RP if possible (or fallback to CPU in the encoder factory logic if needed)
            # Here we just use the helper from mamba_visual_encoder or models.utils
            from models.utils import recurrence_plot_gpu
            # per-channel average
            rp = recurrence_plot_gpu(tokens_for_rp.reshape(B*W*F, L)).reshape(B*W, F, L, L).mean(dim=1)
            rp = rp.view(B, W, L, L)
            
        k = torchF.normalize(vproj(visual(rp).mean(dim=1)), dim=1)
        
        loss = clip_contrastive_loss(q, k)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()); n += 1
    return total / max(1, n)


def _measure_inference(visual, patch_len, device, batch_size=64, num_windows=4):
    """Measures ms/batch and peak memory for the visual encoder ONLY."""
    rp = torch.randn(batch_size, num_windows, patch_len, patch_len).to(device)
    visual.eval()
    
    # Warmup
    for _ in range(5):
        _ = visual(rp)
    
    torch.cuda.reset_peak_memory_stats(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(50):
        _ = visual(rp)
    end_event.record()
    
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event) / 50.0
    mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # MB
    
    return ms, mem


def _probe_evaluate(encoder, visual, data_dir, dataset_csv, horizon, device, probe_epochs=30):
    module = TimeSeriesDataModule(
        dataset_name=dataset_csv, data_dir=str(data_dir),
        batch_size=64, val_batch_size=64,
        num_workers=0, pin_memory=False,
        normalize=True, train=True, val=False, test=True,
        sample_size=(96, 0, horizon),
    )
    module.setup()
    if not module.train_loaders: return {}
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
            # Compute RP for visual
            tokens = tokenize_sequence(x, token_size=visual.patch_len)
            B, W, L, F = tokens.shape
            tokens_for_rp = tokens.permute(0, 1, 3, 2).reshape(B * W, F, L)
            from models.utils import recurrence_plot_gpu
            rp = recurrence_plot_gpu(tokens_for_rp.reshape(B*W*F, L)).reshape(B*W, F, L, L).mean(dim=1)
            rp = rp.view(B, W, L, L)
            zv = visual(rp).mean(dim=1)  # Pool across windows: (B, W, d) → (B, d)
            return torch.cat([ze, zv], dim=1)

    def _collect(loader):
        zs, ys = [], []
        for b in loader:
            zs.append(_embed(b))
            y = b[1].to(device).float() if len(b) > 1 else b[0].to(device).float()
            ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    Z_tr, Y_tr = _collect(train_loader)
    
    feat_dim = Z_tr.shape[1]
    probe = nn.Linear(feat_dim, horizon).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    
    for _ in range(probe_epochs):
        probe.train()
        perm = torch.randperm(Z_tr.shape[0], device=device)
        for i in range(0, Z_tr.shape[0], 64):
            idx = perm[i:i+64]
            z_b = Z_tr[idx]; y_b = Y_tr[idx]
            if y_b.ndim == 3: y_b = y_b[:, :horizon, 0]
            elif y_b.ndim == 2: y_b = y_b[:, :horizon]
            if y_b.shape[1] < horizon: continue
            loss = torchF.mse_loss(probe(z_b), y_b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    results = {"mse": float("nan"), "mae": float("nan")}
    if test_loader:
        Z_test, Y_test = _collect(test_loader)
        probe.eval()
        with torch.no_grad():
            pred = probe(Z_test)
            if Y_test.ndim == 3: Y_test = Y_test[:, :horizon, 0]
            elif Y_test.ndim == 2: Y_test = Y_test[:, :horizon]
            if Y_test.shape[1] >= horizon:
                results = {
                    "mse": torchF.mse_loss(pred, Y_test).item(),
                    "mae": (pred - Y_test).abs().mean().item(),
                }

    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(True)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--results_dir", type=Path, default=src_dir.parent / "results" / "ablation_H")
    p.add_argument("--data_dir", type=Path, default=src_dir.parent / "ICML_datasets")
    p.add_argument("--pretrain_data_dir", type=Path, default=src_dir.parent / "data")
    p.add_argument("--variants", nargs="+", default=ENCODER_VARIANTS)
    p.add_argument("--patch_lens", nargs="+", type=int, default=PATCH_LENGTHS)
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
        num_workers=int(config.data.get("num_workers", 4)),
        pin_memory=bool(config.data.get("pin_memory", True)),
        seed=args.seed,
    )

    rows: List[Dict] = []

    for variant in args.variants:
        for l in args.patch_lens:
            print(f"\n{'='*60}\nEncoder: {variant}  l={l}\n{'='*60}")
            tu.set_seed(args.seed)

            # Build temporal encoder (fixed)
            model_cfg = dict(config.model)
            model_cfg["input_dim"] = l
            encoder = tu.build_encoder_from_config(model_cfg).to(device)
            
            # Build visual encoder (variant)
            visual = VisualEncoderFactory.build(variant, patch_len=l, d_model=128).to(device)
            
            proj = build_projection_head(encoder).to(device)
            vproj = build_projection_head(visual).to(device)

            params = (list(encoder.parameters()) + list(visual.parameters()) 
                      + list(proj.parameters()) + list(vproj.parameters()))
            optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

            # Pre-train
            for ep in range(args.train_epochs):
                loss = _clip_train_epoch(
                    encoder, visual, proj, vproj, train_loader, optimizer, device,
                    noise_std=float(config.training.get("noise_std", 0.01)),
                )
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(f"  epoch {ep+1}/{args.train_epochs}  loss={loss:.4f}")

            # Performance metrics
            ms, mem = _measure_inference(visual, l, device)
            params_count = sum(p.numel() for p in visual.parameters())
            print(f"  Speed: {ms:.2f} ms/batch  Mem: {mem:.1f} MB  Params: {params_count:,}")

            # Probe evaluation
            for ds in PROBE_DATASETS:
                m = _probe_evaluate(encoder, visual, args.data_dir, ds, HORIZON, device, args.probe_epochs)
                if not m or "mse" not in m:
                    print(f"  {ds}  SKIPPED (no results)")
                    continue
                rows.append({
                    "encoder": variant, "l": l, "dataset": ds.replace(".csv", ""),
                    "mse": f"{m['mse']:.4f}", "mae": f"{m['mae']:.4f}",
                    "ms_batch": f"{ms:.2f}", "mem_mb": f"{mem:.1f}", "params": params_count
                })
                print(f"  {ds}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    out_csv = args.results_dir / "ablation_H_results.csv"
    fieldnames = ["encoder", "l", "dataset", "mse", "mae", "ms_batch", "mem_mb", "params"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
