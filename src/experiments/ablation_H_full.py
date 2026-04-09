"""
Ablation H — Full Validation: Visual Encoder Architecture
==========================================================
Proper re-run of Ablation H with:
  - Full LOTSA pre-training (50 epochs, Nano model)
  - All 9 ICML benchmark datasets
  - All 4 horizons: H ∈ {96, 192, 336, 720}
  - 3 scientifically interesting variants only:
      cnn           — surprisingly strong CNN baseline (local filters on RP)
      rp_ss2d_2     — proposed method (2-scan bidirectional Mamba)
      upper_tri_diag — RP-geometry-native (exploits symmetry, reads anti-diagonals as lag tokens)

The original Ablation H used only 20 training epochs and 3 datasets (ETTm1, Weather,
Traffic at H=96), which is insufficient to draw conclusions. This script uses the
same evaluation protocol as the main results (probe_lotsa_checkpoint.py).

Protocol
--------
For each variant:
  1. Pre-train temporal encoder (MambaEncoder, Nano size) + visual encoder (variant)
     on full LOTSA corpus using CLIP-style contrastive loss.
  2. Freeze both encoders.
  3. For each ICML dataset × horizon, train a linear head (probe_epochs epochs).
  4. Report MSE / MAE. Save per-variant CSV.

Usage
-----
  python src/experiments/ablation_H_full.py \\
      --config src/configs/lotsa_nano.yaml \\
      --data_dir ICML_datasets \\
      --results_dir results/ablation_H_full \\
      --train_epochs 50 \\
      --probe_epochs 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
root_dir = src_dir.parent
for p in (src_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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
from models.utils import recurrence_plot_gpu
from time_series_loader import TimeSeriesDataModule

# ── constants ─────────────────────────────────────────────────────────────────

VARIANTS = ["cnn", "rp_ss2d_2", "upper_tri_diag"]

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
    "exchange_rate.csv", "solar_AL.txt",
]
HORIZONS = [96, 192, 336, 720]


# ── RP helper for factory visual encoders ────────────────────────────────────

def _compute_rp_for_visual(x: torch.Tensor, patch_len: int) -> torch.Tensor:
    """Tokenise x and compute per-patch RPs.

    Args:
        x: (B, 1, L) univariate series
    Returns:
        rp: (B, W, patch_len, patch_len) recurrence plots
    """
    tokens = tokenize_sequence(x, token_size=patch_len)   # (B, W, L, F)
    B, W, L, F = tokens.shape
    tokens_for_rp = tokens.permute(0, 1, 3, 2).reshape(B * W * F, L)
    rp = recurrence_plot_gpu(tokens_for_rp)                # (B*W*F, L, L)
    rp = rp.reshape(B * W, F, L, L).mean(dim=1)           # (B*W, L, L) — avg channels
    return rp.view(B, W, L, L)


# ── training ─────────────────────────────────────────────────────────────────

def _clip_train_epoch(encoder, visual, proj, vproj, loader, optimizer, device,
                      noise_std: float = 0.01) -> float:
    encoder.train(); visual.train(); proj.train(); vproj.train()
    total, n = 0.0, 0

    for batch in loader:
        if isinstance(batch, dict) and "target" in batch:
            seq = batch["target"].to(device).float()
            if "lengths" in batch:
                seq = seq[:, :int(batch["lengths"].max().item())]
        elif isinstance(batch, (tuple, list)):
            seq = batch[0].to(device).float()
        else:
            seq = batch.to(device).float()

        seq = prepare_sequence(seq)
        x_q = reshape_multivariate_series(seq)                        # (B, 1, L)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))

        q = F.normalize(proj(encoder(x_q)), dim=1)

        with torch.no_grad():
            rp = _compute_rp_for_visual(x_k, visual.patch_len)       # (B, W, l, l)

        k = F.normalize(vproj(visual(rp).mean(dim=1)), dim=1)

        loss = clip_contrastive_loss(q, k)
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(visual.parameters()) +
            list(proj.parameters()) + list(vproj.parameters()), 1.0
        )
        optimizer.step()
        total += float(loss.item()); n += 1

    return total / max(1, n)


def pretrain(
    encoder: nn.Module,
    visual: nn.Module,
    config,
    device: torch.device,
    train_epochs: int,
    checkpoint_dir: Path,
) -> None:
    """CLIP pre-training on LOTSA. Saves best encoder checkpoints."""
    d_model = int(config.model.get("model_dim", 128))
    proj = build_projection_head(encoder, output_dim=d_model).to(device)
    vproj = build_projection_head(visual, output_dim=d_model).to(device)

    params = (list(encoder.parameters()) + list(visual.parameters()) +
              list(proj.parameters()) + list(vproj.parameters()))
    optimizer = torch.optim.AdamW(params, lr=float(config.training.get("learning_rate", 1e-3)),
                                  weight_decay=float(config.training.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_epochs,
        eta_min=float(config.training.get("min_lr", 1e-6))
    )

    train_loader, _ = build_time_series_dataloaders(
        data_dir=str(root_dir / "data"),
        dataset_name=config.data.get("dataset_name", ""),
        dataset_type=config.data.get("dataset_type", "lotsa"),
        batch_size=int(config.data.get("batch_size", 256)),
        num_workers=int(config.data.get("num_workers", 4)),
        pin_memory=bool(config.data.get("pin_memory", True)),
        seed=42,
        cronos_kwargs=dict(config.data.get("cronos_kwargs", {})),
    )

    noise_std = float(config.training.get("noise_std", 0.01))
    best_loss = float("inf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, train_epochs + 1):
        loss = _clip_train_epoch(encoder, visual, proj, vproj, train_loader,
                                 optimizer, device, noise_std)
        scheduler.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{train_epochs}  loss={loss:.4f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        if loss < best_loss:
            best_loss = loss
            torch.save(encoder.state_dict(), checkpoint_dir / "time_series_best.pt")
            torch.save(visual.state_dict(), checkpoint_dir / "visual_encoder_best.pt")

    print(f"  Pre-training done. Best loss: {best_loss:.4f}")


# ── probe evaluation ──────────────────────────────────────────────────────────

def _embed_all(encoder, visual, loader, device) -> tuple:
    """Collect embeddings and targets for a full loader."""
    zs, ys = [], []
    for batch in loader:
        x = batch[0].to(device).float()
        x = reshape_multivariate_series(prepare_sequence(x))  # (B*C, 1, L)
        with torch.no_grad():
            ze = encoder(x)
            rp = _compute_rp_for_visual(x, visual.patch_len)  # (B*C, W, l, l)
            zv = visual(rp).mean(dim=1)                        # (B*C, d)
        zs.append(torch.cat([ze, zv], dim=1))
        y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
        if y.ndim == 3:
            B, L, Cy = y.shape
            y = y.permute(0, 2, 1).reshape(B * Cy, L)
        ys.append(y)
    if not zs:
        return None, None
    return torch.cat(zs), torch.cat(ys)


def probe_one_dataset(
    encoder: nn.Module,
    visual: nn.Module,
    data_dir: Path,
    dataset_csv: str,
    horizons: List[int],
    device: torch.device,
    probe_epochs: int = 20,
    batch_size: int = 64,
) -> Dict[int, Dict[str, float]]:
    """Linear probe for one dataset across all horizons."""
    resolved_dir = str(data_dir)
    for candidate in data_dir.rglob(dataset_csv):
        resolved_dir = str(candidate.parent)
        break

    max_horizon = max(horizons)
    try:
        module = TimeSeriesDataModule(
            dataset_name=dataset_csv, data_dir=resolved_dir,
            batch_size=batch_size, val_batch_size=batch_size,
            num_workers=0, pin_memory=False, normalize=True,
            train=True, val=False, test=True,
            sample_size=(96, 0, max_horizon), scaler_type="standard",
        )
        module.setup()
    except Exception as exc:
        print(f"    [SKIP] {dataset_csv} — {exc}")
        return {}

    if not module.train_loaders:
        print(f"    [SKIP] {dataset_csv} — no train loader")
        return {}

    encoder.eval(); visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    Z_tr, Y_tr = _embed_all(encoder, visual, module.train_loaders[0], device)
    if Z_tr is None:
        return {}

    feat_dim = Z_tr.shape[1]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    results: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        probe = nn.Linear(feat_dim, H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        for _ in range(probe_epochs):
            probe.train()
            perm = torch.randperm(Z_tr.shape[0], device=device)
            for i in range(0, Z_tr.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                z_b = Z_tr[idx]
                y_b = Y_tr[idx, :H] if Y_tr.ndim == 2 else Y_tr[idx, :H, 0]
                if y_b.shape[-1] < H:
                    continue
                loss = F.mse_loss(probe(z_b), y_b)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if test_loader:
            Z_te, Y_te = _embed_all(encoder, visual, test_loader, device)
            if Z_te is None:
                results[H] = {"mse": float("nan"), "mae": float("nan")}
                continue
            probe.eval()
            with torch.no_grad():
                y_eval = Y_te[:, :H] if Y_te.ndim == 2 else Y_te[:, :H, 0]
                if y_eval.shape[-1] < H:
                    results[H] = {"mse": float("nan"), "mae": float("nan")}
                    continue
                pred = probe(Z_te)
                results[H] = {
                    "mse": F.mse_loss(pred, y_eval).item(),
                    "mae": (pred - y_eval).abs().mean().item(),
                }
        else:
            results[H] = {"mse": float("nan"), "mae": float("nan")}

    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(True)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation H — full validation")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_nano.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path,
                   default=root_dir / "results" / "ablation_H_full")
    p.add_argument("--checkpoint_base", type=Path,
                   default=root_dir / "checkpoints" / "ablation_H_nano")
    p.add_argument("--train_epochs", type=int, default=50)
    p.add_argument("--probe_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_pretrain", action="store_true",
                   help="Skip pre-training, load existing checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device   : {device}")
    print(f"Variants : {args.variants}")
    print(f"Datasets : {len(ICML_DATASETS)} ICML datasets")
    print(f"Horizons : {args.horizons}")
    print(f"Epochs   : pretrain={args.train_epochs}  probe={args.probe_epochs}")

    patch_len = int(config.model.get("input_dim", 64))
    d_model = int(config.model.get("model_dim", 128))

    all_rows: List[Dict] = []

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"Variant: {variant}  (patch_len={patch_len})")
        print(f"{'='*60}", flush=True)

        tu.set_seed(args.seed)

        # Build models
        model_cfg = dict(config.model)
        model_cfg["input_dim"] = patch_len
        encoder = tu.build_encoder_from_config(model_cfg).to(device)
        visual = VisualEncoderFactory.build(variant, patch_len=patch_len, d_model=d_model).to(device)

        enc_params = sum(p.numel() for p in encoder.parameters())
        vis_params = sum(p.numel() for p in visual.parameters())
        print(f"  Encoder params: {enc_params:,}  Visual params: {vis_params:,}", flush=True)

        checkpoint_dir = args.checkpoint_base / variant

        # Pre-train
        if not args.skip_pretrain:
            print(f"\n  Pre-training on LOTSA ({args.train_epochs} epochs)...")
            pretrain(encoder, visual, config, device, args.train_epochs, checkpoint_dir)
        else:
            # Load existing checkpoint
            enc_ckpt = checkpoint_dir / "time_series_best.pt"
            vis_ckpt = checkpoint_dir / "visual_encoder_best.pt"
            if enc_ckpt.exists() and vis_ckpt.exists():
                encoder.load_state_dict(torch.load(enc_ckpt, map_location=device))
                visual.load_state_dict(torch.load(vis_ckpt, map_location=device))
                print(f"  Loaded checkpoint from {checkpoint_dir}", flush=True)
            else:
                print(f"  [WARN] No checkpoint found at {checkpoint_dir}, using random weights")

        # Linear probe on all ICML datasets × horizons
        print(f"\n  Linear probe evaluation...")
        for ds_name in ICML_DATASETS:
            ds_tag = ds_name.replace(".csv", "").replace(".txt", "")
            print(f"\n  Dataset: {ds_tag}", flush=True)
            results = probe_one_dataset(
                encoder=encoder, visual=visual,
                data_dir=args.data_dir, dataset_csv=ds_name,
                horizons=args.horizons, device=device,
                probe_epochs=args.probe_epochs, batch_size=args.batch_size,
            )
            for H, m in results.items():
                mse, mae = m["mse"], m["mae"]
                print(f"    H={H:3d}  MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
                all_rows.append({
                    "variant": variant, "dataset": ds_tag,
                    "horizon": H, "mse": f"{mse:.4f}", "mae": f"{mae:.4f}",
                })

        # Save per-variant CSV
        variant_csv = args.results_dir / f"{variant}_results.csv"
        variant_rows = [r for r in all_rows if r["variant"] == variant]
        # Add mean per dataset
        out_rows = []
        for ds_tag in [r["dataset"] for r in variant_rows if r not in out_rows]:
            ds_rows = [r for r in variant_rows if r["dataset"] == ds_tag]
            if ds_rows and ds_tag not in [r["dataset"] for r in out_rows]:
                out_rows.extend(ds_rows)
                valid = [r for r in ds_rows if r["mse"] != "nan"]
                if valid:
                    mean_mse = sum(float(r["mse"]) for r in valid) / len(valid)
                    mean_mae = sum(float(r["mae"]) for r in valid) / len(valid)
                    out_rows.append({
                        "variant": variant, "dataset": ds_tag,
                        "horizon": "mean", "mse": f"{mean_mse:.4f}", "mae": f"{mean_mae:.4f}",
                    })
        with variant_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["variant", "dataset", "horizon", "mse", "mae"])
            writer.writeheader(); writer.writerows(out_rows)
        print(f"\n  Saved: {variant_csv}", flush=True)

    # Save combined CSV
    combined_csv = args.results_dir / "ablation_H_full_results.csv"
    with combined_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "dataset", "horizon", "mse", "mae"])
        writer.writeheader(); writer.writerows(all_rows)
    print(f"\nAll results saved to {combined_csv}")


if __name__ == "__main__":
    main()
