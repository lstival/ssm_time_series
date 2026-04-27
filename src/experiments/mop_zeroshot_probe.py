"""
Phase 3: Zero-Shot Inference using MoP and Greedy Scheduling.
"""

from __future__ import annotations

import argparse
import sys
import time
import csv
from pathlib import Path
from typing import Dict

import torch

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
    p = argparse.ArgumentParser("MoP Zero-Shot Probe")
    p.add_argument("--mop_checkpoint", type=Path, required=True, help="Path to MoP tuning checkpoint")
    p.add_argument("--base_checkpoint_dir", type=Path, required=True, help="Path to frozen SimCLR/encoder checkpoint")
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "ablation_best.yaml")
    p.add_argument("--data_dir", type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path, default=root_dir / "results" / "mop_tuning")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_prompts", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--context_length", type=int, default=336)
    return p.parse_args()


def build_test_loaders(data_dir: Path, context_length: int, max_horizon: int, batch_size: int):
    test_loaders_map = {}
    for ds_name in ICML_DATASETS:
        ds_tag = ds_name.replace(".csv", "").replace(".txt", "")
        resolved_dir = str(data_dir)
        for candidate in data_dir.rglob(ds_name):
            resolved_dir = str(candidate.parent)
            break
            
        try:
            module = TimeSeriesDataModule(
                dataset_name=ds_name,
                data_dir=resolved_dir,
                batch_size=batch_size,
                val_batch_size=batch_size,
                num_workers=0,
                pin_memory=False,
                normalize=True,
                train=False,
                val=False,
                test=True,
                sample_size=(context_length, 0, max_horizon),
                scaler_type="standard"
            )
            module.setup()
            if module.test_loaders:
                test_loaders_map[ds_tag] = module.test_loaders[0]
        except Exception as exc:
            print(f"Skipping {ds_tag}: {exc}")
    return test_loaders_map


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    
    print("Loading encoders...")
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)
    
    # Load base encoders
    enc_path = args.base_checkpoint_dir / "time_series_best.pt"
    if not enc_path.exists(): enc_path = args.base_checkpoint_dir / "time_series_encoder.pt"
    vis_path = args.base_checkpoint_dir / "visual_encoder_best.pt"
    if not vis_path.exists(): vis_path = args.base_checkpoint_dir / "visual_encoder.pt"
    
    encoder.load_state_dict(torch.load(enc_path, map_location=device).get("model_state_dict", torch.load(enc_path, map_location=device)))
    visual.load_state_dict(torch.load(vis_path, map_location=device).get("model_state_dict", torch.load(vis_path, map_location=device)))
    
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    mop_model = MoPForecastModel(
        encoder=encoder,
        visual_encoder=visual,
        input_dim=enc_dim * 2,
        hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts,
        horizons=HORIZONS,
        target_features=1,
        freeze_encoders=True
    ).to(device)
    
    print(f"Loading MoP checkpoint: {args.mop_checkpoint}")
    ckpt = torch.load(args.mop_checkpoint, map_location=device)
    mop_model.load_state_dict(ckpt["mop_model"])
    mop_model.eval()
    
    max_horizon = max(HORIZONS)
    loaders = build_test_loaders(args.data_dir, args.context_length, max_horizon, args.batch_size)
    
    results = []
    
    print("\n--- Zero-Shot MoP Evaluation (Greedy Scheduling) ---")
    
    with torch.no_grad():
        for ds_tag, loader in loaders.items():
            print(f"Evaluating {ds_tag}...")
            
            # evaluate each horizon individually
            for H in HORIZONS:
                all_preds, all_trues = [], []
                
                for batch in loader:
                    x_raw = batch[0].to(device).float() # (B, ctx, C)
                    y_raw = batch[1].to(device).float() # (B, max_h, C)
                    
                    if y_raw.shape[1] < H:
                        continue
                        
                    # reshape to channel-wise
                    B, L, C = x_raw.shape
                    
                    # Instead of B*C, we reshape appropriately
                    x_ch = x_raw.permute(0, 2, 1).reshape(B*C, 1, L)
                    
                    # Greedy prediction using the frozen exact head if available or recursive composition
                    pred_ch = mop_model.greedy_predict(x_ch, H, args.context_length)
                    # shape is (B*C, H, 1)
                    
                    y_target = y_raw[:, :H, :].permute(0, 2, 1).reshape(B*C, H, 1)
                    
                    all_preds.append(pred_ch)
                    all_trues.append(y_target)
                    
                if not all_preds:
                    continue
                    
                pt = torch.cat(all_preds)
                tt = torch.cat(all_trues)
                
                mse = torch.mean((pt - tt)**2).item()
                mae = torch.mean(torch.abs(pt - tt)).item()
                
                print(f"  {ds_tag} H={H}: MSE={mse:.4f} MAE={mae:.4f}")
                results.append({"dataset": ds_tag, "horizon": H, "mse": mse, "mae": mae})

    out_csv = args.results_dir / "mop_zeroshot_results.csv"
    with open(out_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nDone. Results saved to {out_csv}")

if __name__ == "__main__":
    main()
