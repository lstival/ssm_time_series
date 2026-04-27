"""
Phase 2: MoP Tuning (Supervised Head & Module in-Prompt Training).

This script tunes the MoP (Module of Prompts) and 4 prediction heads
over a frozen pretrained encoder (SimCLR Nano) using the combined
LOTSA, Chronos, and Local datasets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
root_dir = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from dataloaders.lotsa_loader import build_lotsa_dataloaders
from dataloaders.concat_loader import build_dataset_loader_list
from dataloaders.utils import discover_dataset_files

# For now we use the same ICML local datasets for local data blending
LOCAL_DATASETS = "ICML_datasets"
LOCAL_DATASET_BASENAMES = {
    "ETTh1.csv",
    "ETTh2.csv",
    "ETTm1.csv",
    "ETTm2.csv",
    "weather.csv",
    "electricity.csv",
    "traffic.csv",
}


def parse_args():
    p = argparse.ArgumentParser("MoP Tuning (Phase 2)")
    p.add_argument("--checkpoint_dir", type=Path, required=True, help="Path to frozen SimCLR/encoder checkpoint")
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "ablation_best.yaml")
    p.add_argument("--results_dir", type=Path, default=root_dir / "results" / "mop_tuning")
    
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_prompts", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--context_length", type=int, default=336)
    p.add_argument("--max_horizon", type=int, default=720)
    
    # Dataset mixing fractions
    p.add_argument("--batches_per_epoch", type=int, default=1000)
    return p.parse_args()


def load_encoders(checkpoint_dir: Path, config: Any, device: torch.device):
    # Depending on SimCLR or another arch, load appropriately
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)

    # Use _best.pt or latest depending on what exists
    enc_path = checkpoint_dir / "time_series_best.pt"
    if not enc_path.exists():
        enc_path = checkpoint_dir / "time_series_encoder.pt"
    vis_path = checkpoint_dir / "visual_encoder_best.pt"
    if not vis_path.exists():
        vis_path = checkpoint_dir / "visual_encoder.pt"

    print(f"Loading encoder from: {enc_path}")
    enc_state = torch.load(enc_path, map_location=device)
    encoder.load_state_dict(enc_state.get("model_state_dict", enc_state.get("model_state", enc_state)))
    
    print(f"Loading visual from : {vis_path}")
    vis_state = torch.load(vis_path, map_location=device)
    visual.load_state_dict(vis_state.get("model_state_dict", vis_state.get("model_state", vis_state)))

    return encoder, visual

def prepare_batch(batch, source_type, ctx_len, max_h, device):
    """Normalizes batch format to x=(B, C, L), y=(B, C, max_h)"""
    if source_type in ["lotsa", "chronos"]:
        # batch["target"] is (B, L_total, 1) where L_total = ctx_len + max_h
        t = batch["target"].to(device).float()
        if t.shape[1] < ctx_len + max_h:
            return None, None
        
        t = t.transpose(1, 2) # (B, 1, L_total)
        x = t[:, :, :ctx_len]
        y = t[:, :, ctx_len:ctx_len+max_h]
        return x, y
        
    elif source_type == "local":
        # local custom datasets return (x, y, x_mark, y_mark)
        # x is (B, ctx_len, C), y is (B, max_h, C)
        x = batch[0].to(device).float().transpose(1, 2)
        y = batch[1].to(device).float().transpose(1, 2)
        return x, y
        
    return None, None

def train_mop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    
    # We combine representation dimensions for MoP
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    mop_model = MoPForecastModel(
        encoder=encoder,
        visual_encoder=visual,
        input_dim=enc_dim * 2,
        hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts,
        horizons=[96, 192, 336, 720],
        target_features=1,
        freeze_encoders=True
    ).to(device)
    
    # Loaders
    print("Setting up combined dataloaders (LOTSA + Local)...")
    combined_loaders = []
    
    # 1. LOTSA
    lotsa_train, _ = build_lotsa_dataloaders(
        context_length=args.context_length + args.max_horizon,  # Need enough for future exactly
        batch_size=args.batch_size,
        two_views=False
    )
    combined_loaders.append(("lotsa", iter(lotsa_train), lotsa_train))
    
    # 2. Local loaders
    local_path = str(root_dir / LOCAL_DATASETS)
    if Path(local_path).exists():
        discovered_local = discover_dataset_files(local_path)
        local_dataset_files = {
            rel_path: abs_path
            for rel_path, abs_path in discovered_local.items()
            if Path(rel_path).name in LOCAL_DATASET_BASENAMES
        }
        dls = build_dataset_loader_list(
            local_path,
            batch_size=args.batch_size,
            sample_size=(args.context_length, 0, args.max_horizon),
            dataset_files=local_dataset_files,
        )
        for dl in dls:
            if dl.train is not None:
                combined_loaders.append(("local", iter(dl.train), dl.train))
    
    print(f"Total active loaders: {len(combined_loaders)}")
    
    # Optimizer focused solely on MoP and heads
    mop_params = list(mop_model.mop.parameters()) + list(mop_model.heads.parameters())
    optimizer = optim.AdamW(mop_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    
    print("\nStarting MoP Tuning (Phase 2)...")
    
    for epoch in range(1, args.epochs + 1):
        mop_model.train()
        total_loss = 0.0
        n_valid = 0
        t0 = time.time()
        
        # Round Robin through loaders
        loader_idx = 0
        
        for step in range(args.batches_per_epoch):
            # Fetch batch securely
            source_type, data_iter, dataloader = combined_loaders[loader_idx]
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                combined_loaders[loader_idx] = (source_type, data_iter, dataloader)
                batch = next(data_iter)
                
            loader_idx = (loader_idx + 1) % len(combined_loaders)
            
            x, y = prepare_batch(batch, source_type, args.context_length, args.max_horizon, device)
            if x is None or y is None: 
                continue
                
            # Randomly select one horizon to optimize for this batch to save memory/compute,
            # or optimize all. We'll optimize randomly sampled valid horizons for each batch.
            h_target = mop_model.horizons[torch.randint(0, len(mop_model.horizons), (1,)).item()]
            if h_target > y.shape[2]:
                continue
                
            y_target = y[:, :, :h_target]
            
            pred = mop_model(x, h_target) # (B, H, C)
            # y_target is (B, C, H), we want to compare with (B*C, H, 1)
            B, C, H = y_target.shape
            y_target_ch = y_target.permute(0, 2, 1).reshape(B*C, H, 1)
            
            loss = F.mse_loss(pred, y_target_ch)
            
            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(mop_params, 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_valid += 1
                
        scheduler.step()
        
        avg_loss = total_loss / max(1, n_valid)
        print(f"Epoch {epoch:2d}/{args.epochs} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Time: {time.time()-t0:.1f}s")
        
        # Save latest
        torch.save({
            "mop_model": mop_model.state_dict(),
            "epoch": epoch,
            "args": args
        }, args.results_dir / "mop_latest.pt")
        
    print("Done MoP tuning.")

if __name__ == "__main__":
    args = parse_args()
    train_mop(args)
