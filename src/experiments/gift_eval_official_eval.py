"""GIFT-Eval official evaluation — NRMSE and SMAPE in raw (denormalised) space.

Matches the metric definitions used by SEMPO and Chronos in their GIFT-Eval
comparison table (NRMSE = RMSE / mean|y_true|, SMAPE as in the M-competition).

Supports two modes:
  --mode zeroshot  — frozen encoder + MoP zero-shot checkpoint
  --mode fewshot   — frozen encoder + fine-tuned per-dataset MoP checkpoint

Outputs a CSV with one row per (dataset, horizon) and a summary table.

Usage:
    python gift_eval_official_eval.py \\
        --encoder_name   clip_mini_alldata \\
        --checkpoint_dir checkpoints/clip_mini_alldata/ts_clip_mini_alldata_20260503_210737 \\
        --mop_checkpoint results/moms_clip_mini_alldata/mop_zeroshot_clip_mini_alldata_checkpoint.pt \\
        --config         src/configs/lotsa_clip_mini_alldata.yaml \\
        --mode           zeroshot \\
        --results_dir    results/gift_eval_official/clip_mini_alldata_zeroshot
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent
root_dir   = src_dir.parent
for p in (src_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from dataloaders.gift_eval_loader import (
    load_gift_eval_hf, GiftEvalDataset, ALL_GIFT_SSL_SUBSETS,
)
from torch.utils.data import DataLoader


HORIZONS = [96, 192, 336, 720]

# Map Parquet subset name → (display name, native pred_length)
# pred_length = None means use len(future_value) natively
GIFT_SUBSETS = ALL_GIFT_SSL_SUBSETS


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(args, device: torch.device):
    config = tu.load_config(args.config)
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual  = tu.build_visual_encoder_from_config(config.model).to(device)

    ckpt_dir = Path(args.checkpoint_dir)
    for name in ("time_series_best.pt", "time_series_encoder.pt", "time_series_last.pt"):
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            encoder.load_state_dict(state.get("model_state_dict", state), strict=False)
            break
    for name in ("visual_encoder_best.pt", "visual_encoder.pt", "visual_encoder_last.pt"):
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            visual.load_state_dict(state.get("model_state_dict", state), strict=False)
            break

    ckpt     = torch.load(args.mop_checkpoint, map_location=device)
    mop_args = ckpt["args"]

    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 96))
    fusion  = getattr(mop_args, "fusion_mode", "concat")
    in_dim  = enc_dim * 2 if fusion == "concat" else enc_dim

    model = MoPForecastModel(
        encoder=encoder,
        visual_encoder=visual,
        input_dim=in_dim,
        hidden_dim=getattr(mop_args, "hidden_dim", 512),
        num_prompts=getattr(mop_args, "num_prompts", 16),
        horizons=HORIZONS,
        target_features=1,
        freeze_encoders=True,
        norm_mode=getattr(mop_args, "norm_mode", "revin"),
        head_type=getattr(mop_args, "head_type", "linear"),
        use_ln_head=getattr(mop_args, "use_ln_head", False),
        residual_head=getattr(mop_args, "residual_head", False),
        temperature=getattr(mop_args, "temperature", 1.0),
        scale_cond=getattr(mop_args, "scale_cond", False),
        learnable_scale=getattr(mop_args, "learnable_scale", False),
        dropout=0.0,
        fusion_mode=fusion,
    ).to(device)
    model.load_state_dict(ckpt["mop_model"])
    model.eval()
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def nrmse_smape_raw(pred_raw: np.ndarray, true_raw: np.ndarray):
    """
    pred_raw, true_raw: (N, H) arrays in original scale.

    NRMSE = RMSE / mean(|true|)   — matches Chronos/SEMPO definition
    SMAPE = mean(2|p-t| / (|p|+|t|+eps))  — symmetric MAPE, decimal scale
    """
    mse   = np.mean((pred_raw - true_raw) ** 2)
    rmse  = np.sqrt(mse)
    denom = np.mean(np.abs(true_raw))
    nrmse = rmse / (denom + 1e-8)

    smape = np.mean(
        2.0 * np.abs(pred_raw - true_raw) /
        (np.abs(pred_raw) + np.abs(true_raw) + 1e-8)
    )
    return float(nrmse), float(smape)


# ── Per-subset evaluation ─────────────────────────────────────────────────────

def evaluate_subset(
    model: MoPForecastModel,
    subset_name: str,
    horizon: int,
    context_length: int,
    batch_size: int,
    device: torch.device,
) -> dict | None:
    """Evaluate on one subset at one horizon. Returns None if horizon exceeds native pred_len."""
    hf_ds = load_gift_eval_hf(subset_name, split="train", force_offline=True)
    native_pred_len = len(hf_ds[0]["future_value"])

    if horizon > native_pred_len:
        return None  # skip horizons longer than the benchmark's native one

    ds = GiftEvalDataset(
        hf_ds,
        context_length=context_length,
        prediction_length=horizon,  # evaluate at this specific horizon
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_pred_raw, all_true_raw = [], []
    with torch.no_grad():
        for batch in loader:
            x   = batch["target"].to(device).float()          # (B, L, 1)
            y   = batch["future"].to(device).float()          # (B, H, 1)
            mu  = batch["mu"].to(device).float()              # (B,)
            sig = batch["sigma"].to(device).float()           # (B,)

            B, L, C = x.shape
            x_in = x.permute(0, 2, 1)                         # (B, 1, L)

            pred_norm = model.greedy_predict(x_in, horizon, context_length)
            # pred_norm: (B, H, 1)  — in z-score space (model uses RevIN internally,
            # so output is already in input scale — no further denorm needed for RevIN)
            # BUT our loader normalises inputs: raw = norm * sigma + mu
            # model.greedy_predict with norm_mode='revin' does its own norm/denorm
            # on the z-score-normalised input, so the output is still in z-score space.
            # We must undo the loader's normalisation to get raw scale.
            mu3  = mu.view(B, 1, 1)
            sig3 = sig.view(B, 1, 1)
            pred_raw = pred_norm * sig3 + mu3               # (B, H, 1)
            true_raw = y         * sig3 + mu3               # (B, H, 1)

            all_pred_raw.append(pred_raw.squeeze(-1).cpu().numpy())   # (B, H)
            all_true_raw.append(true_raw.squeeze(-1).cpu().numpy())

    if not all_pred_raw:
        return None

    pred_np = np.concatenate(all_pred_raw, axis=0)   # (N, H)
    true_np = np.concatenate(all_true_raw, axis=0)

    nrmse, smape = nrmse_smape_raw(pred_np, true_np)
    return {
        "subset":        subset_name,
        "horizon":       horizon,
        "native_pred_len": native_pred_len,
        "n_windows":     len(pred_np),
        "NRMSE":         nrmse,
        "SMAPE":         smape,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("GIFT-Eval NRMSE/SMAPE Evaluation")
    p.add_argument("--encoder_name",    required=True)
    p.add_argument("--checkpoint_dir",  type=Path, required=True)
    p.add_argument("--mop_checkpoint",  type=Path, required=True)
    p.add_argument("--config",          type=Path, required=True)
    p.add_argument("--mode",            choices=["zeroshot", "fewshot"], default="zeroshot")
    p.add_argument("--results_dir",     type=Path,
                   default=Path("results/gift_eval_official"))
    p.add_argument("--context_length",  type=int, default=336)
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--device",          type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subsets",         nargs="+", default=GIFT_SUBSETS)
    p.add_argument("--horizons",        nargs="+", type=int, default=HORIZONS)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.encoder_name} [{args.mode}]")
    model = load_model(args, device)

    all_results = []
    print(f"\n{'Subset':<40} {'H':>4}  {'NRMSE':>8}  {'SMAPE':>8}  {'N':>6}")
    print("-" * 75)

    for subset in args.subsets:
        for H in args.horizons:
            try:
                res = evaluate_subset(
                    model, subset, H,
                    args.context_length, args.batch_size, device,
                )
                if res is None:
                    continue
                all_results.append(res)
                print(f"{subset:<40} {H:>4}  {res['NRMSE']:>8.4f}  {res['SMAPE']:>8.4f}  {res['n_windows']:>6}")
            except Exception as e:
                print(f"{subset:<40} {H:>4}  ERROR: {e}")
                import traceback; traceback.print_exc()

    # Save CSV
    if all_results:
        out_csv = args.results_dir / f"gift_eval_{args.encoder_name}_{args.mode}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults → {out_csv}")

        # Summary: avg NRMSE and SMAPE per subset (across horizons)
        import pandas as pd
        df = pd.DataFrame(all_results)
        summary = df.groupby("subset")[["NRMSE", "SMAPE"]].mean().reset_index()
        print("\n--- Summary (avg across horizons) ---")
        print(f"{'Dataset':<40}  {'NRMSE':>8}  {'SMAPE':>8}")
        print("-" * 60)
        for _, row in summary.iterrows():
            print(f"{row['subset']:<40}  {row['NRMSE']:>8.4f}  {row['SMAPE']:>8.4f}")
        print(f"\nOverall avg NRMSE: {df['NRMSE'].mean():.4f}")
        print(f"Overall avg SMAPE: {df['SMAPE'].mean():.4f}")


if __name__ == "__main__":
    main()
