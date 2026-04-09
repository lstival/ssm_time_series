"""
ICML Finetuning vs Supervised Training Comparison
===================================================
Three experimental conditions evaluated on the ICML benchmark datasets:

  finetune   — Load LOTSA pretrained encoders, unfreeze them, jointly train
               encoders + MLP head end-to-end on ICML train split (MSE loss).

  supervised — Random-init encoders (same architecture), train encoders + MLP
               head end-to-end on ICML train split (MSE loss). No pretraining.

  (probe)    — Reference baseline: frozen LOTSA encoders + linear head only
               (run separately via probe_lotsa_checkpoint.py).

Goal: show that LOTSA pretraining + finetuning outperforms training from scratch
on the same (limited) ICML data, demonstrating the value of self-supervised
pretraining on a large diverse corpus.

Usage
-----
  # Finetuning (start from LOTSA checkpoint)
  python src/experiments/icml_finetune_vs_supervised.py \\
      --mode finetune \\
      --checkpoint_dir checkpoints/lotsa_ablation_best/ts_encoder_lotsa_ablation_best_20260403_1213 \\
      --config src/configs/lotsa_ablation_best.yaml \\
      --data_dir ICML_datasets \\
      --results_dir results/icml_finetune

  # Supervised from scratch (no pretraining)
  python src/experiments/icml_finetune_vs_supervised.py \\
      --mode supervised \\
      --config src/configs/lotsa_ablation_best.yaml \\
      --data_dir ICML_datasets \\
      --results_dir results/icml_supervised
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
from torch.utils.data import DataLoader, ConcatDataset

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
root_dir = src_dir.parent
for p in (src_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import training_utils as tu
from util import prepare_sequence, reshape_multivariate_series
from time_series_loader import TimeSeriesDataModule
from models.dual_forecast import DualEncoderForecastMLP

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
    "exchange_rate.csv", "solar_AL.txt",
]
HORIZONS = [96, 192, 336, 720]


# ── encoder helpers ───────────────────────────────────────────────────────────

def build_encoders(config, device: torch.device):
    """Build fresh random-init encoders."""
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)
    return encoder, visual


def load_pretrained_encoders(checkpoint_dir: Path, config, device: torch.device):
    """Load LOTSA pretrained encoders (unfrozen for finetuning)."""
    encoder, visual = build_encoders(config, device)

    enc_state = torch.load(checkpoint_dir / "time_series_best.pt", map_location=device)
    vis_state = torch.load(checkpoint_dir / "visual_encoder_best.pt", map_location=device)

    encoder.load_state_dict(
        enc_state.get("model_state_dict", enc_state.get("model_state", enc_state))
    )
    visual.load_state_dict(
        vis_state.get("model_state_dict", vis_state.get("model_state", vis_state))
    )
    print(f"Loaded encoder       : {checkpoint_dir / 'time_series_best.pt'}")
    print(f"Loaded visual encoder: {checkpoint_dir / 'visual_encoder_best.pt'}")
    return encoder, visual


# ── embedding helper ──────────────────────────────────────────────────────────

def _embed_batch(encoder, visual, x_raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Embed raw batch, treating each channel as independent univariate sample.

    Args:
        x_raw: (B, L, C)
    Returns:
        (B*C, 2D) embeddings
    """
    seq = prepare_sequence(x_raw.to(device).float())  # (B, L, C)
    x = reshape_multivariate_series(seq)              # (B*C, 1, L)
    ze = encoder(x)   # (B*C, D)
    zv = visual(x)    # (B*C, D)
    return torch.cat([ze, zv], dim=1)                 # (B*C, 2D)


# ── data loading ──────────────────────────────────────────────────────────────

def build_icml_loaders(
    data_dir: Path,
    context_length: int,
    max_horizon: int,
    batch_size: int,
    scaler_type: str = "standard",
):
    """Build combined train loader over all ICML datasets, and per-dataset test loaders."""
    train_loaders, test_loaders_map = [], {}

    for ds_name in ICML_DATASETS:
        ds_tag = ds_name.replace(".csv", "").replace(".txt", "")
        resolved_dir = str(data_dir)
        for candidate in data_dir.rglob(ds_name):
            resolved_dir = str(candidate.parent)
            break

        # Train loader
        try:
            module_tr = TimeSeriesDataModule(
                dataset_name=ds_name,
                data_dir=resolved_dir,
                batch_size=batch_size,
                val_batch_size=batch_size,
                num_workers=0,
                pin_memory=False,
                normalize=True,
                train=True,
                val=False,
                test=False,
                sample_size=(context_length, 0, max_horizon),
                scaler_type=scaler_type,
            )
            module_tr.setup()
            if module_tr.train_loaders:
                train_loaders.append(module_tr.train_loaders[0])
                print(f"  Train: {ds_tag} — {len(module_tr.train_loaders[0].dataset)} samples", flush=True)
        except Exception as exc:
            print(f"  [SKIP train] {ds_tag}: {exc}")

        # Test loader
        try:
            module_te = TimeSeriesDataModule(
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
                scaler_type=scaler_type,
            )
            module_te.setup()
            if module_te.test_loaders:
                test_loaders_map[ds_tag] = module_te.test_loaders[0]
        except Exception as exc:
            print(f"  [SKIP test] {ds_tag}: {exc}")

    return train_loaders, test_loaders_map


# ── training loop ─────────────────────────────────────────────────────────────

def train(
    encoder: nn.Module,
    visual: nn.Module,
    mlp: DualEncoderForecastMLP,
    train_loaders: List[DataLoader],
    device: torch.device,
    epochs: int,
    context_length: int,
    max_horizon: int,
    lr: float,
    encoder_lr_scale: float,
    checkpoint_path: Path,
    warmup_epochs: int = 3,
):
    """Joint training: encoders + MLP, all unfrozen.

    Stability notes:
    - AMP (float16) disabled: Mamba SSM backward is unstable in float16.
    - embed_norm: LayerNorm applied to embeddings before MLP — critical for
      supervised mode where random-init encoders produce arbitrary-scale outputs.
    - Warmup LR: encoder LR starts at lr*1e-3 and ramps over warmup_epochs.
    - Checkpoint saved after epoch 1 regardless of loss, so torch.load never fails.
    """
    encoder.train()
    visual.train()
    mlp.train()

    enc_dim = mlp.input_dim  # 2 * embedding_dim
    embed_norm = nn.LayerNorm(enc_dim).to(device)

    enc_base_lr = lr * encoder_lr_scale
    all_params = list(encoder.parameters()) + list(visual.parameters()) + list(mlp.parameters()) + list(embed_norm.parameters())
    optimizer = torch.optim.AdamW([
        {"params": list(encoder.parameters()) + list(visual.parameters()), "lr": enc_base_lr},
        {"params": list(mlp.parameters()) + list(embed_norm.parameters()), "lr": lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    nan_batches = 0
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Linear warmup for encoder LR: start at 0.1% and ramp to full over warmup_epochs
        if epoch <= warmup_epochs:
            warmup_scale = epoch / warmup_epochs
            optimizer.param_groups[0]["lr"] = enc_base_lr * warmup_scale

        epoch_loss, n_batches = 0.0, 0

        for loader in train_loaders:
            for batch in loader:
                x = batch[0].to(device).float()  # (B, ctx, C)
                y = batch[1].to(device).float()  # (B, H, C)
                B, L, C = x.shape
                Hy = y.shape[1]

                x_ch = x.permute(0, 2, 1).reshape(B * C, L, 1)
                y_ch = y.permute(0, 2, 1).reshape(B * C, Hy)
                y_target = y_ch[:, :max_horizon]
                if y_target.shape[1] < max_horizon:
                    continue

                # float32 forward — no AMP (Mamba SSM NaN in float16 backward)
                z = _embed_batch(encoder, visual, x_ch, device)  # (B*C, 2D)

                if not torch.isfinite(z).all():
                    nan_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                z = embed_norm(z)                   # stabilise scale for random-init encoders
                pred = mlp(z)[:, :, 0]              # (B*C, max_H)
                loss = criterion(pred, y_target)

                if not torch.isfinite(loss):
                    nan_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = (time.time() - t0) / 60
        nan_info = f"  [NaN skipped: {nan_batches}]" if nan_batches > 0 else ""
        print(f"  Epoch {epoch:3d}/{epochs} — Loss: {avg_loss:.4f}  LR_head: {scheduler.get_last_lr()[-1]:.6f}  ({elapsed:.1f} min){nan_info}", flush=True)

        # Always save after epoch 1 so checkpoint file always exists
        if avg_loss < best_loss or epoch == 1:
            if n_batches > 0:  # only save if we had valid batches
                best_loss = min(avg_loss, best_loss)
                torch.save({
                    "encoder": encoder.state_dict(),
                    "visual": visual.state_dict(),
                    "mlp": mlp.state_dict(),
                    "embed_norm": embed_norm.state_dict(),
                }, checkpoint_path)
                print(f"    → Saved best (loss={best_loss:.4f})", flush=True)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return best_loss, embed_norm


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    encoder: nn.Module,
    visual: nn.Module,
    mlp: DualEncoderForecastMLP,
    test_loaders_map: Dict,
    device: torch.device,
    horizons: List[int],
    results_dir: Path,
    mode: str,
    embed_norm: Optional[nn.Module] = None,
):
    encoder.eval()
    visual.eval()
    mlp.eval()
    if embed_norm is not None:
        embed_norm.eval()
    max_horizon = max(horizons)

    rows = []
    print(f"\n{'='*60}")
    print(f"Evaluation — mode: {mode}")
    print(f"{'='*60}")

    for ds_tag, test_loader in test_loaders_map.items():
        print(f"\n  Dataset: {ds_tag}")
        all_preds = {H: [] for H in horizons}
        all_trues = {H: [] for H in horizons}

        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device).float()  # (B, ctx, C)
                y = batch[1].to(device).float()  # (B, H, C)
                B, L, C = x.shape
                Hy = y.shape[1]

                x_ch = x.permute(0, 2, 1).reshape(B * C, L, 1)
                z = _embed_batch(encoder, visual, x_ch, device)
                if embed_norm is not None:
                    z = embed_norm(z)
                pred_full = mlp(z)[:, :, 0]  # (B*C, max_H)
                y_ch = y.permute(0, 2, 1).reshape(B * C, Hy)

                for H in horizons:
                    if H > Hy or H > pred_full.shape[1]:
                        continue
                    all_preds[H].append(pred_full[:, :H])
                    all_trues[H].append(y_ch[:, :H])

        for H in horizons:
            if not all_preds[H]:
                print(f"    H={H:3d}  [no data]")
                continue
            pred_t = torch.cat(all_preds[H])
            true_t = torch.cat(all_trues[H])
            mse = F.mse_loss(pred_t, true_t).item()
            mae = (pred_t - true_t).abs().mean().item()
            print(f"    H={H:3d}  MSE={mse:.4f}  MAE={mae:.4f}")
            rows.append({"dataset": ds_tag, "horizon": H, "mse": f"{mse:.4f}", "mae": f"{mae:.4f}"})

    # Add mean per dataset
    out_rows = []
    datasets_seen = []
    for row in rows:
        if row["dataset"] not in datasets_seen:
            datasets_seen.append(row["dataset"])
    for ds in datasets_seen:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        out_rows.extend(ds_rows)
        mean_mse = sum(float(r["mse"]) for r in ds_rows) / len(ds_rows)
        mean_mae = sum(float(r["mae"]) for r in ds_rows) / len(ds_rows)
        out_rows.append({"dataset": ds, "horizon": "mean", "mse": f"{mean_mse:.4f}", "mae": f"{mean_mae:.4f}"})

    out_csv = results_dir / f"icml_{mode}_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nResults saved to {out_csv}")
    return out_rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ICML finetuning vs supervised comparison")
    p.add_argument("--mode", choices=["finetune", "supervised"], required=True,
                   help="finetune: start from LOTSA checkpoint; supervised: train from scratch")
    p.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="Required for --mode finetune")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_ablation_best.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path,
                   default=root_dir / "results" / "icml_comparison")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate for MLP head (encoder LR = lr * encoder_lr_scale)")
    p.add_argument("--encoder_lr_scale", type=float, default=0.1,
                   help="Encoder LR multiplier relative to MLP LR (default 0.1 = 10x smaller)")
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--context_length", type=int, default=96)
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    p.add_argument("--scaler_type", type=str, default="standard")
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "finetune" and args.checkpoint_dir is None:
        raise ValueError("--checkpoint_dir is required for --mode finetune")

    config = tu.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {args.mode}")
    print(f"Epochs : {args.epochs}")
    print(f"LR     : {args.lr} (encoder LR = {args.lr * args.encoder_lr_scale:.2e})")

    t_start = time.time()
    max_horizon = max(args.horizons)

    # Build encoders
    if args.mode == "finetune":
        encoder, visual = load_pretrained_encoders(args.checkpoint_dir, config, device)
    else:
        encoder, visual = build_encoders(config, device)
        print("Encoders initialised from random weights (supervised baseline).")

    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    combined_dim = enc_dim * 2

    mlp = DualEncoderForecastMLP(
        input_dim=combined_dim,
        hidden_dim=args.mlp_hidden_dim,
        horizons=args.horizons,
        target_features=1,
        dropout=0.1,
    ).to(device)

    # Load ICML data
    print(f"\nLoading ICML datasets from {args.data_dir} ...")
    train_loaders, test_loaders_map = build_icml_loaders(
        data_dir=args.data_dir,
        context_length=args.context_length,
        max_horizon=max_horizon,
        batch_size=args.batch_size,
        scaler_type=args.scaler_type,
    )

    if not train_loaders:
        raise RuntimeError("No ICML training data found. Check --data_dir.")

    checkpoint_path = args.results_dir / f"best_{args.mode}.pt"

    print(f"\n{'='*60}")
    print(f"Training — mode: {args.mode}")
    print(f"{'='*60}")
    _, embed_norm = train(
        encoder=encoder,
        visual=visual,
        mlp=mlp,
        train_loaders=train_loaders,
        device=device,
        epochs=args.epochs,
        context_length=args.context_length,
        max_horizon=max_horizon,
        lr=args.lr,
        encoder_lr_scale=args.encoder_lr_scale,
        checkpoint_path=checkpoint_path,
    )

    # Reload best weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    visual.load_state_dict(ckpt["visual"])
    mlp.load_state_dict(ckpt["mlp"])
    if "embed_norm" in ckpt:
        embed_norm.load_state_dict(ckpt["embed_norm"])

    evaluate(
        encoder=encoder,
        visual=visual,
        mlp=mlp,
        test_loaders_map=test_loaders_map,
        device=device,
        horizons=args.horizons,
        results_dir=args.results_dir,
        mode=args.mode,
        embed_norm=embed_norm,
    )

    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
