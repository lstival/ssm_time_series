"""
Flexible MoP Evaluation Script (Zero-Shot).
Loads a mop_flex checkpoint and evaluates on benchmark datasets.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
root_dir = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from time_series_loader import TimeSeriesDataModule

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv"
]
HORIZONS = [96, 192, 336, 720]

def parse_args():
    p = argparse.ArgumentParser("Flexible MoP Evaluation")
    p.add_argument("--mop_checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint_dir", type=Path, required=True, help="Encoder checkpoints")
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "ablation_best.yaml")
    p.add_argument("--data_dir", type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path, default=root_dir / "results" / "mop_eval")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def load_encoders(checkpoint_dir: Path, config: Any, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)
    enc_path = checkpoint_dir / "time_series_best.pt"
    if not enc_path.exists(): enc_path = checkpoint_dir / "time_series_encoder.pt"
    vis_path = checkpoint_dir / "visual_encoder_best.pt"
    if not vis_path.exists(): vis_path = checkpoint_dir / "visual_encoder.pt"
    encoder.load_state_dict(torch.load(enc_path, map_location=device).get("model_state_dict", {}), strict=False)
    visual.load_state_dict(torch.load(vis_path, map_location=device).get("model_state_dict", {}), strict=False)
    return encoder, visual

def main():
    args = parse_args()
    device = torch.device(args.device)
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading MoP checkpoint: {args.mop_checkpoint}")
    ckpt = torch.load(args.mop_checkpoint, map_location=device)
    mop_args = ckpt["args"]
    
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    
    model = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=enc_dim * 2, hidden_dim=mop_args.hidden_dim, num_prompts=mop_args.num_prompts,
        horizons=HORIZONS, target_features=1, freeze_encoders=True,
        norm_mode=mop_args.norm_mode, head_type=mop_args.head_type, use_ln_head=mop_args.use_ln_head,
        residual_head=mop_args.residual_head, temperature=mop_args.temperature,
        scale_cond=mop_args.scale_cond, learnable_scale=mop_args.learnable_scale, dropout=mop_args.dropout
    ).to(device)
    model.load_state_dict(ckpt["mop_model"])
    model.eval()
    
    results = []
    print("\n--- Flexible MoP Evaluation (Zero-Shot) ---")
    
    with torch.no_grad():
        for ds_name in ICML_DATASETS:
            ds_tag = ds_name.replace(".csv", "")
            resolved = str(args.data_dir)
            for c in args.data_dir.rglob(ds_name):
                resolved = str(c.parent)
                break
                
            module = TimeSeriesDataModule(
                dataset_name=ds_name, data_dir=resolved, batch_size=args.batch_size,
                normalize=True, train=False, test=True,
                sample_size=(mop_args.context_length, 0, max(HORIZONS))
            )
            module.setup()
            if not module.test_loaders: continue
            loader = module.test_loaders[0]
            
            print(f"Evaluating {ds_tag}...")
            for H in HORIZONS:
                all_preds, all_trues = [], []
                for batch in loader:
                    x_raw = batch[0].to(device).float().transpose(1, 2)
                    y_raw = batch[1].to(device).float()
                    if y_raw.shape[1] < H: continue
                    
                    # Greedy predict handles internal normalization if configured
                    B, C, L = x_raw.shape
                    x_ch = x_raw.reshape(B * C, 1, L)
                    
                    pred = model.greedy_predict(x_ch, H, mop_args.context_length)
                    y_true = y_raw[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)
                    
                    all_preds.append(pred)
                    all_trues.append(y_true)
                
                if not all_preds: continue
                pt, tt = torch.cat(all_preds), torch.cat(all_trues)
                mse = torch.mean((pt - tt)**2).item()
                mae = torch.mean(torch.abs(pt - tt)).item()
                print(f"  {ds_tag} H={H}: MSE={mse:.4f} MAE={mae:.4f}")
                results.append({"dataset": ds_tag, "horizon": H, "mse": mse, "mae": mae})

    out_csv = args.results_dir / f"eval_{args.mop_checkpoint.stem}.csv"
    with open(out_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDone. Results saved to {out_csv}")

if __name__ == "__main__":
    main()
