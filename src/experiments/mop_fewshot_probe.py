"""
Few-Shot Evaluation of MoP.
==========================
Following the SEMPO methodology for Few-Shot:
1. Freeze the backbone (SimCLR nano encoder).
2. Initialize the MoP + Prediction Heads (optionally loading a pre-trained MoP checkpoint).
3. Fine-Tune ONLY the MoP + Heads using a small fraction (e.g., 5%) of the target dataset's training split.
4. Evaluate zero-shot (or simply test inference) on the target dataset's completely hidden test split.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

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
from util import prepare_sequence, reshape_multivariate_series

PROBE_DATASETS: List[str] = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv"
]
HORIZONS: List[int] = [96, 192, 336, 720]


def parse_args():
    p = argparse.ArgumentParser("MoP Few-Shot Fine-Tuning and Evaluation")
    p.add_argument("--base_checkpoint_dir", type=Path, required=True, help="Path to frozen SimCLR/encoder")
    p.add_argument("--mop_checkpoint", type=Path, default=None, help="Path to pre-trained MoP (optional, limits to few-shot transfer)")
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "lotsa_simclr_bimodal_nano.yaml")
    p.add_argument("--data_dir", type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path, default=root_dir / "results" / "mop_tuning")
    
    p.add_argument("--few_shot_fraction", type=float, default=0.05, help="Fraction of train data to use (e.g. 0.05 for 5%)")
    p.add_argument("--finetune_epochs", type=int, default=10, help="Epochs to fine-tune MoP on the 5% fraction")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_prompts", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--context_length", type=int, default=336)
    p.add_argument("--seed", type=int, default=42)
    
    return p.parse_args()


def get_fresh_mop(args, config, encoder, visual, device):
    """Creates a fresh MoP, optionally loading weights if passed."""
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
    
    if args.mop_checkpoint is not None and args.mop_checkpoint.exists():
        ckpt = torch.load(args.mop_checkpoint, map_location=device)
        mop_model.load_state_dict(ckpt["mop_model"])
        
    return mop_model


def load_encoders(checkpoint_dir: Path, config, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)

    enc_ckpt = checkpoint_dir / "time_series_best.pt"
    if not enc_ckpt.exists(): enc_ckpt = checkpoint_dir / "time_series_encoder.pt"
    vis_ckpt = checkpoint_dir / "visual_encoder_best.pt"
    if not vis_ckpt.exists(): vis_ckpt = checkpoint_dir / "visual_encoder.pt"

    if enc_ckpt.exists():
        enc_state = torch.load(enc_ckpt, map_location=device)
        encoder.load_state_dict(enc_state.get("model_state_dict", enc_state.get("model_state", enc_state)))
    if vis_ckpt.exists():
        vis_state = torch.load(vis_ckpt, map_location=device)
        visual.load_state_dict(vis_state.get("model_state_dict", vis_state.get("model_state", vis_state)))

    encoder.eval()
    visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    return encoder, visual


def main():
    args = parse_args()
    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Few-Shot MoP Probing")
    print(f"Fraction: {args.few_shot_fraction*100}%")
    print(f"Fine-tune epochs: {args.finetune_epochs}")
    
    encoder, visual = load_encoders(args.base_checkpoint_dir, config, device)
    
    results = []
    rng = torch.Generator()
    rng.manual_seed(args.seed)
    
    for ds_csv in PROBE_DATASETS:
        ds_tag = ds_csv.replace(".csv", "").replace(".txt", "")
        print(f"\n{'='*50}\nEvaluating Few-Shot on {ds_tag}\n{'='*50}")
        
        # Resolve dataset
        resolved_dir = str(args.data_dir)
        for candidate in args.data_dir.rglob(ds_csv):
            resolved_dir = str(candidate.parent)
            break
            
        module = TimeSeriesDataModule(
            dataset_name=ds_csv,
            data_dir=resolved_dir,
            batch_size=args.batch_size,
            val_batch_size=args.batch_size,
            num_workers=0,
            pin_memory=False,
            normalize=True,
            train=True,
            val=False,
            test=True,
            sample_size=(args.context_length, 0, max(HORIZONS)),
            scaler_type="standard"
        )
        try:
            module.setup()
        except:
            print(f"Skipping {ds_tag}: setup failed.")
            continue
            
        train_loader = module.train_loaders[0] if module.train_loaders else None
        test_loader = module.test_loaders[0] if module.test_loaders else None
        
        if train_loader is None or test_loader is None:
            continue
            
        # 1. Instantiate fresh MoP model (and load pre-trained if provided)
        mop_model = get_fresh_mop(args, config, encoder, visual, device)
        mop_params = list(mop_model.mop.parameters()) + list(mop_model.heads.parameters())
        optimizer = optim.AdamW(mop_params, lr=args.lr, weight_decay=1e-4)
        
        # 2. Extract Data for Few-Shot into memory to easily subsample and train 
        # (Alternatively just iterate over loader exactly `n_keep` samples)
        # We will collect everything to easily sample exactly the fraction
        print("Extracting train set into memory to isolate 5%...")
        all_x, all_y = [], []
        for batch in train_loader:
            x_b = batch[0].float() # (B, L, C)
            y_b = batch[1].float() # (B, max_H, C)
            all_x.append(x_b)
            all_y.append(y_b)
        
        tx = torch.cat(all_x, dim=0)
        ty = torch.cat(all_y, dim=0)
        
        n_total = tx.shape[0]
        n_keep = max(1, int(n_total * args.few_shot_fraction))
        idx = torch.randperm(n_total, generator=rng)[:n_keep]
        
        fx = tx[idx]
        fy = ty[idx]
        print(f"Few-Shot subset: {n_keep} samples (total: {n_total})")
        
        # 3. Fine-Tune MoP on the 5%
        print("Fine-tuning MoP + Heads...")
        mop_model.train()
        for ep in range(args.finetune_epochs):
            ep_loss = 0.0
            perm = torch.randperm(n_keep)
            for i in range(0, n_keep, args.batch_size):
                b_idx = perm[i : i + args.batch_size]
                bx = fx[b_idx].to(device)
                by = fy[b_idx].to(device)
                
                # Transform to (B, C, L) for the MoP model inputs format
                bx_t = bx.transpose(1, 2)
                
                # Randomize horizon optimization over all available to generic fix all heads
                h_target = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
                if h_target > by.shape[1]:
                    continue
                    
                target_H = by[:, :h_target, :] # (B, H, C)
                pred = mop_model(bx_t, h_target) # (B, H, C)  Wait: actually MoP forecast outputs (B*C, H, 1) and reshaped is needed
                
                B, H, C = target_H.shape
                target_H_ch = target_H.permute(0, 2, 1).reshape(B*C, H, 1)
                
                loss = F.mse_loss(pred, target_H_ch)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                
            if ep % 5 == 0 or ep == args.finetune_epochs - 1:
                print(f"  Ep {ep}: loss = {ep_loss:.4f}")
                
        # 4. Final Evaluation (Zero-Shot / Few-Shot on test set)
        print("Evaluating on Test Set...")
        mop_model.eval()
        with torch.no_grad():
            for H in HORIZONS:
                all_preds, all_trues = [], []
                for batch in test_loader:
                    bx = batch[0].to(device).float()
                    by = batch[1].to(device).float()
                    
                    if by.shape[1] < H:
                        continue
                        
                    bx_t = bx.transpose(1, 2)
                    B, L_c, C = bx.shape
                    
                    # We just need to do greedy predict per channel
                    # Greedy predict input shape is (B*C, 1, L)
                    bx_ch = bx_t.reshape(B*C, 1, L_c)
                    
                    pred_ch = mop_model.greedy_predict(bx_ch, H, args.context_length)
                    target_ch = by[:, :H, :].permute(0, 2, 1).reshape(B*C, H, 1)
                    
                    all_preds.append(pred_ch.cpu())
                    all_trues.append(target_ch.cpu())
                    
                if not all_preds:
                    continue
                    
                pt = torch.cat(all_preds)
                tt = torch.cat(all_trues)
                
                mse = torch.mean((pt - tt)**2).item()
                mae = torch.mean(torch.abs(pt - tt)).item()
                
                print(f"    H={H:3d}: MSE={mse:.4f} MAE={mae:.4f}")
                results.append({
                    "dataset": ds_tag,
                    "horizon": H,
                    "mse": mse,
                    "mae": mae
                })

    out_csv = args.results_dir / f"mop_fewshot_{int(args.few_shot_fraction*100)}pct_results.csv"
    with open(out_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nDone! Results saved to {out_csv}")

if __name__ == "__main__":
    main()
