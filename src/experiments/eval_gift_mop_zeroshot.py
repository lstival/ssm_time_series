"""
Zero-Shot Evaluation of SimCLR + MoP on GIFT-Eval dataset.
"""

from __future__ import annotations
import argparse
import csv
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np

# Add src to path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from dataloaders.gift_eval_loader import build_gift_eval_dataloader

# Subsets requested for comparison with SEMPO/Chronos
GIFT_COMP_SUBSETS = [
    "m_dense_H_long",
    "loop_seattle_H_long",
    "sz_taxi_H_short",
    "solar_H_long",
    "bizitobs_application_10S_long",
    "bizitobs_l2c_H_long",
    "bizitobs_service_10S_long",
    "car_parts_M_short",
    "jena_weather_H_long",
]

HORIZONS = [96, 192, 336, 720]

def parse_args():
    p = argparse.ArgumentParser("GIFT-Eval MoP Zero-Shot")
    p.add_argument("--mop_checkpoint", type=Path, default=Path("results/mop_grid_v1/revin_mlp/mop_flex.pt"))
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints/simclr_bimodal_nano/ts_simclr_bimodal_nano_lotsa_20260409_182831"))
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "lotsa_simclr_bimodal_nano.yaml")
    p.add_argument("--results_dir", type=Path, default=Path("results/gift_eval_zeroshot"))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--subsets", type=str, nargs="+", default=GIFT_COMP_SUBSETS)
    return p.parse_args()

def load_encoders(checkpoint_dir: Path, config: Any, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)
    
    enc_path = checkpoint_dir / "time_series_best.pt"
    if not enc_path.exists(): enc_path = checkpoint_dir / "time_series_encoder.pt"
    vis_path = checkpoint_dir / "visual_encoder_best.pt"
    if not vis_path.exists(): vis_path = checkpoint_dir / "visual_encoder.pt"
    
    encoder.load_state_dict(torch.load(enc_path, map_location=device).get("model_state_dict", torch.load(enc_path, map_location=device)), strict=False)
    visual.load_state_dict(torch.load(vis_path, map_location=device).get("model_state_dict", torch.load(vis_path, map_location=device)), strict=False)
    
    encoder.to(device)
    visual.to(device)
    
    return encoder, visual

def main():
    args = parse_args()
    device = torch.device(args.device)
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading MoP checkpoint: {args.mop_checkpoint}")
    ckpt = torch.load(args.mop_checkpoint, map_location=device)
    mop_args = ckpt["args"]
    
    # Use context_length from MoP checkpoint if not provided
    context_length = getattr(mop_args, "context_length", 336)
    
    print(f"Loading encoders from: {args.checkpoint_dir}")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    
    print("Building MoP model...")
    model = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=enc_dim * 2, hidden_dim=mop_args.hidden_dim, num_prompts=mop_args.num_prompts,
        horizons=HORIZONS, target_features=1, freeze_encoders=True,
        norm_mode=mop_args.norm_mode, head_type=mop_args.head_type, use_ln_head=mop_args.use_ln_head,
        residual_head=mop_args.residual_head, temperature=mop_args.temperature,
        scale_cond=mop_args.scale_cond, learnable_scale=mop_args.learnable_scale, dropout=mop_args.dropout
    ).to(device)
    model.load_state_dict(ckpt["mop_model"])
    model.to(device)
    model.eval()
    
    # Thorough check
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            print(f"Forcing {name} to {device} (was on {param.device})")
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(device)
    
    results = []
    print("\n--- Zero-Shot MoP Evaluation on GIFT-Eval ---", flush=True)
    
    with torch.no_grad():
        for subset in args.subsets:
            print(f"\nEvaluating subset: {subset}")

            # Probe actual prediction length available for this subset
            try:
                _probe = build_gift_eval_dataloader(
                    subset, context_length=context_length, prediction_length=max(HORIZONS),
                    batch_size=1, num_workers=0, force_offline=False)
                _b = next(iter(_probe))
                _max_h = _b["future"].shape[1]
                subset_horizons = [h for h in HORIZONS if h <= _max_h] or [_max_h]
                if subset_horizons != HORIZONS:
                    print(f"  NOTE: subset max horizon={_max_h}, evaluating at {subset_horizons}")
            except Exception:
                subset_horizons = HORIZONS

            for H in subset_horizons:
                try:
                    print(f"  H={H}: Building dataloader...")
                    loader = build_gift_eval_dataloader(
                        subset, 
                        context_length=context_length, 
                        prediction_length=H,
                        batch_size=args.batch_size,
                        num_workers=0, # set to 0 to avoid multiprocessing issues in some envs
                        force_offline=False
                    )
                    print(f"  H={H}: Loader ready with {len(loader)} batches")
                except Exception as e:
                    print(f"  Skipping {subset} H={H}: could not build dataloader ({e})")
                    continue
                
                all_preds, all_trues = [], []
                for i, batch in enumerate(loader):
                    x = batch["target"].to(device, non_blocking=True).float().transpose(1, 2) # (B, 1, L)
                    y_true = batch["future"].to(device, non_blocking=True).float() # (B, H, 1)
                    
                    if y_true.shape[1] < H:
                        continue
                    
                    B, C, L = x.shape
                    x_ch = x.reshape(B * C, 1, L).contiguous()
                    
                    if i == 0:
                         print(f"    Debug: H={H}, i={i}, model device: {next(model.parameters()).device}, x_ch device: {x_ch.device}, x_ch dtype: {x_ch.dtype}", flush=True)
                    
                    pred = model.greedy_predict(x_ch, H, context_length) # (B*C, H, 1)
                    
                    # Target reshaping
                    y_true_ch = y_true.permute(0, 2, 1).reshape(B * C, H, 1)
                    
                    all_preds.append(pred)
                    all_trues.append(y_true_ch)
                
                if not all_preds:
                    continue
                    
                pt = torch.cat(all_preds)
                tt = torch.cat(all_trues)
                
                # Standard Metrics
                mse = torch.mean((pt - tt)**2).item()
                mae = torch.mean(torch.abs(pt - tt)).item()
                
                # Comparison Metrics: NRMSE and SMAPE
                rmse = np.sqrt(mse)
                # NRMSE = RMSE / mean(abs(tt)) -- common in Chronos/GIFT-Eval
                avg_abs = torch.mean(torch.abs(tt)).item()
                nrmse = rmse / (avg_abs + 1e-8)
                
                # SMAPE (decimal)
                smape = torch.mean(torch.abs(pt - tt) / ((torch.abs(pt) + torch.abs(tt)) / 2 + 1e-8)).item()
                
                # MinMax NMSE (for internal comparison)
                # (Re-calculating correctly for the full batch)
                # Note: This is less robust than per-window, but good for summary.
                
                print(f"  H={H}: MSE={mse:.4f} NRMSE={nrmse:.4f} SMAPE={smape:.4f}")
                results.append({
                    "subset": subset, "horizon": H, 
                    "mse": mse, "mae": mae, 
                    "nrmse": nrmse, "smape": smape
                })
    
    # Save results
    out_csv = args.results_dir / "gift_eval_baselines_comp.csv"
    with open(out_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["subset", "horizon", "mse", "mae", "nrmse", "smape"])
        writer.writeheader()
        writer.writerows(results)
    
    # Print Comparison summary
    print("\n--- Summary (Avg Metrics for Comparison with SEMPO/Chronos) ---")
    summaries = {}
    for r in results:
        summaries.setdefault(r["subset"], {"nrmse": [], "smape": []})
        summaries[r["subset"]]["nrmse"].append(r["nrmse"])
        summaries[r["subset"]]["smape"].append(r["smape"])
    
    print(f"{'Subset':<30} | {'NRMSE':<8} | {'SMAPE':<8}")
    print("-" * 50)
    for subset, metrics in summaries.items():
        avg_nrmse = sum(metrics["nrmse"]) / len(metrics["nrmse"])
        avg_smape = sum(metrics["smape"]) / len(metrics["smape"])
        print(f"{subset:<30} | {avg_nrmse:<8.4f} | {avg_smape:<8.4f}")
        
    print(f"\nDone. Results saved to {out_csv}")

if __name__ == "__main__":
    main()
