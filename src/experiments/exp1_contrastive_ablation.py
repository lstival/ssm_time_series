"""
EXP-1: Is Contrastive Learning Actually Necessary?
Ablation study to compare temporal-only, multimodal-only, and contrastive multimodal models.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure src is in the python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

import training_utils as tu
from training_utils import set_seed, ExperimentConfig, load_config
from util import (
    run_contrastive_training, 
    run_clip_training,
    default_device,
    prepare_run_directory
)
from models.mamba_encoder import MambaEncoder
from models.mamba_visual_encoder import MambaVisualEncoder
from models.dual_forecast import DualEncoderForecastRegressor, DualEncoderForecastMLP
from supervised_training import ForecastModel
from experiments.silhouette_scorer import compute_silhouette, get_periodic_labels
from time_series_loader import TimeSeriesDataModule

# Constants
DATASETS = ["ETTh1.csv", "electricity.csv", "weather.csv"]
HORIZONS = [96, 192, 336, 720]
BASE_CONFIG_PATH = Path("src/configs/mamba_encoder.yaml")

def build_temporal_model(config, horizon, device):
    encoder = tu.build_encoder_from_config(config.model)
    input_dim = config.model.get('input_dim', 32)
    target_dim = 1 # Assuming univariate or mean-aggregated
    model = ForecastModel(encoder, input_dim=input_dim, target_dim=target_dim, pred_len=horizon)
    return model.to(device)

def build_multimodal_model(config, horizons, device, freeze_encoders=False):
    encoder = tu.build_encoder_from_config(config.model)
    visual_encoder = tu.build_visual_encoder_from_config(config.model, rp_mode="correct")
    
    embedding_dim = config.model.get('embedding_dim', 128)
    head = DualEncoderForecastMLP(
        input_dim=embedding_dim * 2,
        hidden_dim=embedding_dim,
        horizons=horizons,
        target_features=1
    )
    
    model = DualEncoderForecastRegressor(
        encoder=encoder,
        visual_encoder=visual_encoder,
        head=head,
        freeze_encoders=freeze_encoders
    )
    return model.to(device)

def train_supervised(model, train_loader, val_loader, epochs, device, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_x, seq_y = batch[:2]
            seq_x, seq_y = seq_x.to(device).float(), seq_y.to(device).float()
            
            optimizer.zero_grad()
            preds = model(seq_x)
            
            # Match horizons
            if preds.shape[1] > seq_y.shape[1]:
                preds = preds[:, :seq_y.shape[1], :]
            elif seq_y.shape[1] > preds.shape[1]:
                seq_y = seq_y[:, :preds.shape[1], :]
                
            loss = criterion(preds, seq_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader, horizons, device):
    model.eval()
    results = {h: {"mse": 0.0, "mae": 0.0, "count": 0} for h in horizons}
    
    with torch.no_grad():
        for batch in test_loader:
            seq_x, seq_y = batch[:2]
            seq_x, seq_y = seq_x.to(device).float(), seq_y.to(device).float()
            
            preds = model(seq_x)
            
            for h in horizons:
                if h > seq_y.shape[1] or h > preds.shape[1]:
                    continue
                
                h_preds = preds[:, :h, :]
                h_targets = seq_y[:, :h, :]
                
                mse = nn.functional.mse_loss(h_preds, h_targets).item()
                mae = nn.functional.l1_loss(h_preds, h_targets).item()
                
                results[h]["mse"] += mse * seq_x.size(0)
                results[h]["mae"] += mae * seq_x.size(0)
                results[h]["count"] += seq_x.size(0)
                
    final_results = {}
    for h in horizons:
        if results[h]["count"] > 0:
            final_results[h] = {
                "mse": results[h]["mse"] / results[h]["count"],
                "mae": results[h]["mae"] / results[h]["count"]
            }
    return final_results

def get_embeddings(model, loader, device, multimodal=True):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            seq_x = batch[0].to(device).float()
            if multimodal:
                # dual encoder forward without head
                x_transposed = seq_x.transpose(1, 2)
                e1 = model.encoder(x_transposed)
                e2 = model.visual_encoder(x_transposed)
                emb = torch.cat([e1, e2], dim=-1)
            else:
                # single encoder
                if hasattr(model, 'encoder'):
                    x_transposed = seq_x.transpose(1, 2)
                    emb = model.encoder(x_transposed)
                else:
                    # Generic case if model is just encoder wrapper
                    emb = model(seq_x)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0).numpy()

def main(args):
    set_seed(42)
    device = default_device()
    config = tu.load_config(BASE_CONFIG_PATH)
    
    run_dir = prepare_run_directory("results", "exp1_ablation")
    all_results = []
    
    for ds_name in DATASETS:
        print(f"\n--- Processing Dataset: {ds_name} ---")
        # Update config for current dataset
        config.data['dataset_name'] = ds_name
        config.data['data_dir'] = "ICML_datasets"
        
        # tu.prepare_dataloaders expects (config, root)
        train_loader, val_loader = tu.prepare_dataloaders(config, Path.cwd())
        
        # 1. Temporal-Only (Baseline)
        print("\n[Variant 1] Temporal-Only")
        v1_model = build_temporal_model(config, max(HORIZONS), device)
        train_supervised(v1_model, train_loader, val_loader, args.epochs, device)
        v1_metrics = evaluate_model(v1_model, val_loader, HORIZONS, device)
        v1_embs = get_embeddings(v1_model, val_loader, device, multimodal=False)
        # Periodic labels: use batch indices % 24 as a proxy for hourly patterns
        labels = np.arange(len(v1_embs)) % 24 
        v1_sil = compute_silhouette(v1_embs, labels)
        
        for h in HORIZONS:
            all_results.append({
                "Dataset": ds_name, "Variant": "Temporal-Only", "Horizon": h,
                "MSE": v1_metrics[h]["mse"], "MAE": v1_metrics[h]["mae"], "Silhouette": v1_sil
            })

        # 2. Multimodal, No Contrastive
        print("\n[Variant 2] Multimodal, No Contrastive")
        v2_model = build_multimodal_model(config, HORIZONS, device, freeze_encoders=False)
        train_supervised(v2_model, train_loader, val_loader, args.epochs, device)
        v2_metrics = evaluate_model(v2_model, val_loader, HORIZONS, device)
        v2_embs = get_embeddings(v2_model, val_loader, device, multimodal=True)
        v2_sil = compute_silhouette(v2_embs, labels)
        
        for h in HORIZONS:
            all_results.append({
                "Dataset": ds_name, "Variant": "Multimodal-NoContrast", "Horizon": h,
                "MSE": v2_metrics[h]["mse"], "MAE": v2_metrics[h]["mae"], "Silhouette": v2_sil
            })

        # 3. Multimodal, Contrastive Only (Linear Probing)
        print("\n[Variant 3] Multimodal, Contrastive Only (SSL)")
        v3_model = build_multimodal_model(config, HORIZONS, device, freeze_encoders=False)
        # Pretrain encoders using CLIP objective
        ssl_epochs = args.epochs if not args.dry_run else 1
        run_clip_training(
            encoder=v3_model.encoder,
            visual_encoder=v3_model.visual_encoder,
            projection_head=nn.Identity(), # or actual projection
            visual_projection_head=nn.Identity(),
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=run_dir / ds_name / "ssl",
            epochs=ssl_epochs
        )
        # Freeze encoders for linear probing
        for p in v3_model.encoder.parameters():
            p.requires_grad = False
        for p in v3_model.visual_encoder.parameters():
            p.requires_grad = False

        # Linear probe head
        train_supervised(v3_model, train_loader, val_loader, args.epochs, device)
        v3_metrics = evaluate_model(v3_model, val_loader, HORIZONS, device)
        v3_embs = get_embeddings(v3_model, val_loader, device, multimodal=True)
        v3_sil = compute_silhouette(v3_embs, labels)
        
        for h in HORIZONS:
            all_results.append({
                "Dataset": ds_name, "Variant": "Multimodal-SSL-Probe", "Horizon": h,
                "MSE": v3_metrics[h]["mse"], "MAE": v3_metrics[h]["mae"], "Silhouette": v3_sil
            })

        # 4. Multimodal, Contrastive + Supervised (Fine-tune)
        print("\n[Variant 4] Multimodal, Contrastive + Supervised")
        v4_model = build_multimodal_model(config, HORIZONS, device, freeze_encoders=False)
        # Reuse pretrained encoders from V3 if available, or pretrain again
        v4_model.encoder.load_state_dict(v3_model.encoder.state_dict())
        v4_model.visual_encoder.load_state_dict(v3_model.visual_encoder.state_dict())
        
        train_supervised(v4_model, train_loader, val_loader, args.epochs, device)
        v4_metrics = evaluate_model(v4_model, val_loader, HORIZONS, device)
        v4_embs = get_embeddings(v4_model, val_loader, device, multimodal=True)
        v4_sil = compute_silhouette(v4_embs, labels)
        
        for h in HORIZONS:
            all_results.append({
                "Dataset": ds_name, "Variant": "Multimodal-SSL-FineTune", "Horizon": h,
                "MSE": v4_metrics[h]["mse"], "MAE": v4_metrics[h]["mae"], "Silhouette": v4_sil
            })

    # Save all results
    df = pd.DataFrame(all_results)
    output_path = run_dir / "exp1_ablation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nAll experiments completed. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of supervised training epochs per variant")
    parser.add_argument("--dry_run", action="store_true", help="Run a quick test")
    args = parser.parse_args()
    
    if args.dry_run:
        print("Running in DRY RUN mode")
        args.epochs = 1
        DATASETS = ["ETTh1.csv"]
        HORIZONS = [96]
        
    main(args)
