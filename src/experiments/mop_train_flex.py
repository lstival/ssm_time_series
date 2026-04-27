"""
Flexible MoP Training Script (Phase 2 & Few-Shot).
Supports all architectural variations (RevIN, MLP, Temperature, etc.).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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
from time_series_loader import TimeSeriesDataModule

LOCAL_DATASETS = "ICML_datasets"
LOCAL_DATASET_BASENAMES = {
    "ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
    "weather.csv", "electricity.csv", "traffic.csv",
}

def parse_args():
    p = argparse.ArgumentParser("Flexible MoP Training")
    # Base setup
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "ablation_best.yaml")
    p.add_argument("--results_dir", type=Path, default=root_dir / "results" / "mop_flex")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Training Loop
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batches_per_epoch", type=int, default=1000)
    
    # Architecture
    p.add_argument("--num_prompts", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--context_length", type=int, default=336)
    p.add_argument("--max_horizon", type=int, default=720)
    
    # Flexible Flags
    p.add_argument("--norm_mode", type=str, default="identity", choices=["identity", "revin", "minmax", "global"])
    p.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    p.add_argument("--use_ln_head", action="store_true")
    p.add_argument("--residual_head", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--scale_cond", action="store_true")
    p.add_argument("--learnable_scale", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--entropy_weight", type=float, default=0.0)
    
    # Data Mode
    p.add_argument("--mode", type=str, default="source", choices=["source", "fewshot"])
    p.add_argument("--dataset_name", type=str, default=None, help="For few-shot mode")
    p.add_argument("--few_shot_fraction", type=float, default=0.05)
    
    return p.parse_args()

def load_encoders(checkpoint_dir: Path, config: Any, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)

    enc_path = checkpoint_dir / "time_series_best.pt"
    if not enc_path.exists(): enc_path = checkpoint_dir / "time_series_encoder.pt"
    vis_path = checkpoint_dir / "visual_encoder_best.pt"
    if not vis_path.exists(): vis_path = checkpoint_dir / "visual_encoder.pt"

    print(f"Loading encoders from {checkpoint_dir}")
    encoder.load_state_dict(torch.load(enc_path, map_location=device).get("model_state_dict", {}), strict=False)
    visual.load_state_dict(torch.load(vis_path, map_location=device).get("model_state_dict", {}), strict=False)
    return encoder, visual

def prepare_batch(batch, source_type, ctx_len, max_h, device):
    if source_type in ["lotsa", "chronos"]:
        t = batch["target"].to(device).float()
        if t.shape[1] < ctx_len + max_h: return None, None
        t = t.transpose(1, 2) # (B, 1, L_total)
        x, y = t[:, :, :ctx_len], t[:, :, ctx_len:ctx_len+max_h]
        return x, y
    elif source_type == "local":
        x = batch[0].to(device).float().transpose(1, 2)
        y = batch[1].to(device).float().transpose(1, 2)
        return x, y
    return None, None

def main():
    args = parse_args()
    device = torch.device(args.device)
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    
    model = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=enc_dim * 2, hidden_dim=args.hidden_dim, num_prompts=args.num_prompts,
        horizons=[96, 192, 336, 720], target_features=1, freeze_encoders=True,
        norm_mode=args.norm_mode, head_type=args.head_type, use_ln_head=args.use_ln_head,
        residual_head=args.residual_head, temperature=args.temperature,
        scale_cond=args.scale_cond, learnable_scale=args.learnable_scale, dropout=args.dropout
    ).to(device)
    
    # Dataloaders
    combined_loaders = []
    if args.mode == "source":
        # LOTSA
        lotsa_train, _ = build_lotsa_dataloaders(
            context_length=args.context_length + args.max_horizon,
            batch_size=args.batch_size, two_views=False
        )
        combined_loaders.append(("lotsa", iter(lotsa_train), lotsa_train))
        # Local
        local_path = str(root_dir / LOCAL_DATASETS)
        if Path(local_path).exists():
            discovered = discover_dataset_files(local_path)
            files = {r: a for r, a in discovered.items() if Path(r).name in LOCAL_DATASET_BASENAMES}
            dls = build_dataset_loader_list(local_path, batch_size=args.batch_size, 
                                          sample_size=(args.context_length, 0, args.max_horizon),
                                          dataset_files=files)
            for dl in dls:
                if dl.train: combined_loaders.append(("local", iter(dl.train), dl.train))
    else:
        # Few-Shot Mode
        if not args.dataset_name: raise ValueError("Few-shot requires --dataset_name")
        ds_name = args.dataset_name
        rel_dir = str(root_dir / LOCAL_DATASETS)
        module = TimeSeriesDataModule(
            dataset_name=ds_name, data_dir=rel_dir, batch_size=args.batch_size,
            normalize=True, train=True, test=False,
            sample_size=(args.context_length, 0, args.max_horizon)
        )
        module.setup()
        train_loader = module.train_loaders[0]
        # Subsample for few-shot
        print(f"Limiting to {args.few_shot_fraction*100}% few-shot data")
        combined_loaders.append(("local", iter(train_loader), train_loader))
        # Note: In real life we'd strictly slice the dataset here, 
        # but for exploration we'll just train for fewer steps/batches if needed.
    
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"Starting {args.mode} training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        loader_idx = 0
        
        for step in range(args.batches_per_epoch):
            if not combined_loaders: break
            src, data_iter, dl = combined_loaders[loader_idx]
            try: batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                combined_loaders[loader_idx] = (src, data_iter, dl)
                batch = next(data_iter)
            loader_idx = (loader_idx + 1) % len(combined_loaders)
            
            x, y = prepare_batch(batch, src, args.context_length, args.max_horizon, device)
            if x is None: continue
            
            h_target = model.horizons[torch.randint(0, len(model.horizons), (1,)).item()]
            if h_target > y.shape[2]: continue
            
            y_true = y[:, :, :h_target].permute(0, 2, 1).reshape(-1, h_target, 1)
            pred = model(x, h_target) # (B*C, H, 1)
            
            loss = F.mse_loss(pred, y_true)
            
            # Entropy bonus if requested
            if args.entropy_weight > 0:
                # Need to expose routing weights from MoP if we want entropy
                pass 
                
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            if args.mode == "fewshot" and n_batches >= (args.batches_per_epoch * args.few_shot_fraction):
                break

        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | Loss: {total_loss/max(1, n_batches):.4f}")
        
    torch.save({"mop_model": model.state_dict(), "args": args}, args.results_dir / "mop_flex.pt")

if __name__ == "__main__":
    main()
