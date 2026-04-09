"""
Train a forecasting MLP on LOTSA (frozen dual encoders) then zero-shot evaluate
on ICML benchmark datasets.

Pipeline
--------
Stage 1 — MLP training on LOTSA:
  - Load frozen encoder + visual_encoder from --checkpoint_dir (ablation-best)
  - Train DualEncoderForecastMLP (same architecture as production) on LOTSA data
  - Each channel treated as an independent univariate series (matches pre-training)
  - Save MLP weights to --mlp_checkpoint

Stage 2 — Zero-shot evaluation on ICML datasets:
  - Load frozen encoders + trained MLP
  - No fine-tuning: direct inference on unseen datasets
  - Report MSE / MAE per (dataset, horizon), averaged across all channels
  - Each channel evaluated independently (fair to univariate training)

Usage
-----
  python src/experiments/lotsa_forecast_zeroshot.py \
      --checkpoint_dir checkpoints/lotsa_ablation_best/ts_encoder_lotsa_ablation_best_20260403_1213 \
      --config src/configs/lotsa_ablation_best.yaml \
      --results_dir results/lotsa_zeroshot
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
root_dir = src_dir.parent
for p in (src_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import training_utils as tu
from util import prepare_sequence, reshape_multivariate_series
from time_series_loader import TimeSeriesDataModule
from models.dual_forecast import DualEncoderForecastMLP, DualEncoderForecastRegressor
from dataloaders.lotsa_dataset import load_lotsa_datasets
from dataloaders.lotsa_loader import LotsaWindowDataset as LotSAWindowDataset, _lotsa_collate_fn as lotsa_collate_fn


# ── constants ─────────────────────────────────────────────────────────────────

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
    "exchange_rate.csv", "solar_AL.txt",
]
HORIZONS = [96, 192, 336, 720]

LOTSA_DATASETS = [
    "m4_daily", "m4_hourly", "m4_monthly", "m4_yearly", "m4_weekly", "m4_quarterly",
    "exchange_rate", "traffic_hourly", "traffic_weekly", "hospital", "covid_deaths",
    "pedestrian_counts", "nn5_daily_with_missing", "nn5_weekly",
    "monash_m3_monthly", "monash_m3_quarterly", "monash_m3_yearly",
    "taxi_30min", "kdd_cup_2018_with_missing", "oikolab_weather",
    "saugeenday", "us_births", "sunspot_with_missing",
]


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_encoders(checkpoint_dir: Path, config, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)

    enc_state = torch.load(checkpoint_dir / "time_series_best.pt", map_location=device)
    vis_state = torch.load(checkpoint_dir / "visual_encoder_best.pt", map_location=device)

    encoder.load_state_dict(
        enc_state.get("model_state_dict", enc_state.get("model_state", enc_state))
    )
    visual.load_state_dict(
        vis_state.get("model_state_dict", vis_state.get("model_state", vis_state))
    )
    encoder.eval()
    visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    print(f"Loaded encoder      : {checkpoint_dir / 'time_series_best.pt'}")
    print(f"Loaded visual encoder: {checkpoint_dir / 'visual_encoder_best.pt'}")
    return encoder, visual


# ── embedding helper ──────────────────────────────────────────────────────────

def _embed_batch(encoder, visual, x_raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Embed a raw batch, treating each channel as an independent univariate sample.

    Args:
        x_raw: (B, L, C) or (B, L, 1) or (B, L) tensor
    Returns:
        (B*C, 2D) embeddings — one per (sample, channel)
    """
    seq = prepare_sequence(x_raw.to(device).float())  # (B, L, C)
    x = reshape_multivariate_series(seq)              # (B*C, 1, L)
    with torch.no_grad():
        ze = encoder(x)   # (B*C, D)
        zv = visual(x)    # (B*C, D)
    return torch.cat([ze, zv], dim=1)                 # (B*C, 2D)


# ── Stage 1: MLP training on LOTSA ───────────────────────────────────────────

def train_mlp_on_lotsa(
    encoder: nn.Module,
    visual: nn.Module,
    config,
    device: torch.device,
    horizons: List[int],
    mlp_hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    mlp_checkpoint: Path,
    context_length: int = 96,
    cache_dir: str = "/lustre/nobackup/WUR/AIN/stiva001/hf_datasets",
) -> DualEncoderForecastMLP:
    """Train MLP head on LOTSA data with frozen encoders."""

    print("\n" + "="*60)
    print("Stage 1: Training MLP on LOTSA datasets")
    print("="*60)

    # Build LOTSA loaders (same as encoder pre-training, without two_views)
    print("Loading LOTSA datasets...")
    combined_ds = load_lotsa_datasets(
        dataset_names=LOTSA_DATASETS,
        cache_dir=cache_dir,
        normalize_per_series=True,
        force_offline=True,
    )

    max_horizon = max(horizons)
    # We need context + horizon length windows for forecasting
    window_length = context_length + max_horizon

    dataset = LotSAWindowDataset(
        combined_ds,
        context_length=window_length,
        two_views=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lotsa_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"  LOTSA loader: {len(loader)} batches/epoch, window={window_length} (ctx={context_length} + max_horizon={max_horizon})")

    # Infer embedding dim from encoders
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    combined_dim = enc_dim * 2

    mlp = DualEncoderForecastMLP(
        input_dim=combined_dim,
        hidden_dim=mlp_hidden_dim,
        horizons=horizons,
        target_features=1,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        mlp.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in loader:
            # batch["target"]: (B, window_length, 1) from LotSAWindowDataset
            target = batch["target"].to(device).float()  # (B, W, 1)
            B, W, _ = target.shape

            # Split context and future
            x_ctx  = target[:, :context_length, :]   # (B, ctx, 1)
            y_true = target[:, context_length:context_length + max_horizon, 0]  # (B, H)

            # Embed context — (B, 2D) since univariate (C=1)
            z = _embed_batch(encoder, visual, x_ctx, device)  # (B, 2D)

            # Predict all horizons at once
            pred = mlp(z)  # (B, max_horizon, 1)
            pred = pred[:, :, 0]  # (B, max_horizon)

            loss = criterion(pred, y_true)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = (time.time() - t0) / 60

        print(f"  Epoch {epoch:3d}/{epochs} — Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}  ({elapsed:.1f} min)")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state_dict": mlp.state_dict(), "horizons": horizons, "embedding_dim": combined_dim}, mlp_checkpoint)
            print(f"    → Saved best MLP (loss={best_loss:.4f})")

    print(f"\nMLP training complete. Best loss: {best_loss:.4f}")
    print(f"MLP saved to: {mlp_checkpoint}")
    return mlp


# ── Stage 2: Zero-shot evaluation on ICML ────────────────────────────────────

def zeroshot_evaluate(
    encoder: nn.Module,
    visual: nn.Module,
    mlp: DualEncoderForecastMLP,
    data_dir: Path,
    horizons: List[int],
    device: torch.device,
    results_dir: Path,
    context_length: int = 96,
    batch_size: int = 256,
) -> List[Dict]:
    """Zero-shot evaluation: no fine-tuning, direct inference on ICML datasets."""

    print("\n" + "="*60)
    print("Stage 2: Zero-shot evaluation on ICML datasets")
    print("="*60)

    mlp.eval()
    rows = []
    max_horizon = max(horizons)

    for ds_name in ICML_DATASETS:
        ds_tag = ds_name.replace(".csv", "").replace(".txt", "")
        print(f"\n  Dataset: {ds_tag}")

        # Resolve subdirectory
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
                scaler_type="standard",
            )
            module.setup()
        except Exception as exc:
            print(f"    [SKIP] {exc}")
            continue

        if not module.test_loaders:
            print(f"    [SKIP] no test loader")
            continue

        test_loader = module.test_loaders[0]

        # Collect all predictions and targets
        all_preds: Dict[int, List] = {H: [] for H in horizons}
        all_trues: Dict[int, List] = {H: [] for H in horizons}

        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device).float()  # (B, ctx, C)
                y = batch[1].to(device).float()  # (B, horizon, C)
                B, L, C = x.shape

                # Treat each channel independently — matches training distribution
                # x: (B, ctx, C) → (B*C, ctx, 1)
                x_ch = x.permute(0, 2, 1).reshape(B * C, L, 1)
                z = _embed_batch(encoder, visual, x_ch, device)  # (B*C, 2D)

                pred_full = mlp(z)           # (B*C, max_horizon, 1)
                pred_full = pred_full[:, :, 0]  # (B*C, max_horizon)

                # y: (B, max_horizon, C) → (B*C, max_horizon)
                Hy = y.shape[1]
                y_ch = y.permute(0, 2, 1).reshape(B * C, Hy)

                for H in horizons:
                    if H > Hy:
                        continue
                    all_preds[H].append(pred_full[:, :H])
                    all_trues[H].append(y_ch[:, :H])

        for H in horizons:
            if not all_preds[H]:
                print(f"    H={H:3d}  [no data]")
                continue
            pred_t = torch.cat(all_preds[H], dim=0)  # (N*C, H)
            true_t = torch.cat(all_trues[H], dim=0)  # (N*C, H)
            mse = F.mse_loss(pred_t, true_t).item()
            mae = (pred_t - true_t).abs().mean().item()
            print(f"    H={H:3d}  MSE={mse:.4f}  MAE={mae:.4f}")
            rows.append({"dataset": ds_tag, "horizon": H, "mse": f"{mse:.4f}", "mae": f"{mae:.4f}"})

    # Save CSV
    out_csv = results_dir / "lotsa_zeroshot_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP on LOTSA + zero-shot eval on ICML")
    p.add_argument("--checkpoint_dir", type=Path, required=True,
                   help="Dir with time_series_best.pt and visual_encoder_best.pt")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_ablation_best.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path,
                   default=root_dir / "results" / "lotsa_zeroshot")
    p.add_argument("--mlp_checkpoint", type=Path, default=None,
                   help="Where to save the trained MLP (default: results_dir/mlp_best.pt)")
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    p.add_argument("--context_length", type=int, default=96)
    p.add_argument("--mlp_epochs", type=int, default=20)
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cache_dir", type=str,
                   default="/lustre/nobackup/WUR/AIN/stiva001/hf_datasets")
    p.add_argument("--skip_training", action="store_true",
                   help="Skip MLP training and load from --mlp_checkpoint directly")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.mlp_checkpoint is None:
        args.mlp_checkpoint = args.results_dir / "mlp_best.pt"

    config = tu.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Horizons: {args.horizons}")
    print(f"Context length: {args.context_length}")

    t_start = time.time()

    # Load frozen encoders
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)

    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    combined_dim = enc_dim * 2

    if args.skip_training:
        # Load pre-trained MLP
        print(f"\nLoading MLP from {args.mlp_checkpoint}")
        ckpt = torch.load(args.mlp_checkpoint, map_location=device)
        mlp = DualEncoderForecastMLP(
            input_dim=combined_dim,
            hidden_dim=args.mlp_hidden_dim,
            horizons=args.horizons,
            target_features=1,
            dropout=0.1,
        ).to(device)
        mlp.load_state_dict(ckpt["model_state_dict"])
    else:
        # Stage 1: train MLP
        mlp = train_mlp_on_lotsa(
            encoder=encoder,
            visual=visual,
            config=config,
            device=device,
            horizons=args.horizons,
            mlp_hidden_dim=args.mlp_hidden_dim,
            epochs=args.mlp_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            mlp_checkpoint=args.mlp_checkpoint,
            context_length=args.context_length,
            cache_dir=args.cache_dir,
        )
        # Reload best weights
        ckpt = torch.load(args.mlp_checkpoint, map_location=device)
        mlp.load_state_dict(ckpt["model_state_dict"])

    # Stage 2: zero-shot evaluation
    zeroshot_evaluate(
        encoder=encoder,
        visual=visual,
        mlp=mlp,
        data_dir=args.data_dir,
        horizons=args.horizons,
        device=device,
        results_dir=args.results_dir,
        context_length=args.context_length,
        batch_size=args.batch_size,
    )

    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
