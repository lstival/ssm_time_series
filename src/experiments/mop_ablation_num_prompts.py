"""
MoP Ablation — Number of Prompts (LOTSA + Chronos corpus).

Trains one MoP per num_prompts value on the full LOTSA+local combined corpus
(same data the SimCLR-Full encoder was trained on), then evaluates zero-shot
on the 7 ICML benchmark test splits.

Tested values: [2, 4, 8, 16, 32]

The encoder is always frozen. norm_mode='identity' avoids double-normalisation
with the per-series normalisation applied by the combined loader.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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
from dataloaders.local_dataset_loader import build_combined_dataloaders

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
]
HORIZONS = [96, 192, 336, 720]
NUM_PROMPTS_VALUES = [2, 4, 8, 16, 32]


def parse_args():
    p = argparse.ArgumentParser("MoP Ablation — num_prompts on LOTSA+Chronos")
    p.add_argument("--checkpoint_dir", type=Path, required=True,
                   help="SimCLR-Full checkpoint trained on LOTSA+Chronos")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_simclr_full.yaml")
    p.add_argument("--icml_data_dir", type=Path,
                   default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path,
                   default=root_dir / "results" / "mop_ablation_num_prompts")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--context_length", type=int, default=336)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_prompts_values", type=int, nargs="+",
                   default=NUM_PROMPTS_VALUES,
                   help="List of num_prompts to ablate, e.g. --num_prompts_values 4 8 16")
    return p.parse_args()


def resolve_dir(data_dir: Path, ds_name: str) -> str:
    for c in data_dir.rglob(ds_name):
        return str(c.parent)
    return str(data_dir)


def build_icml_test_loaders(icml_data_dir: Path, context_length: int,
                             max_horizon: int, batch_size: int) -> dict:
    loaders = {}
    for ds_name in ICML_DATASETS:
        tag = ds_name.replace(".csv", "")
        resolved = resolve_dir(icml_data_dir, ds_name)
        try:
            module = TimeSeriesDataModule(
                dataset_name=ds_name,
                data_dir=resolved,
                batch_size=batch_size,
                val_batch_size=batch_size,
                num_workers=0,
                pin_memory=False,
                normalize=True,
                train=False, val=False, test=True,
                sample_size=(context_length, 0, max_horizon),
                scaler_type="standard",
            )
            module.setup()
            if module.test_loaders:
                loaders[tag] = module.test_loaders[0]
        except Exception as e:
            print(f"  [WARN] Skipping ICML test {tag}: {e}")
    return loaders


def load_encoders(checkpoint_dir: Path, config, device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual  = tu.build_visual_encoder_from_config(config.model).to(device)

    enc_path = checkpoint_dir / "time_series_best.pt"
    if not enc_path.exists():
        enc_path = checkpoint_dir / "time_series_encoder.pt"
    vis_path = checkpoint_dir / "visual_encoder_best.pt"
    if not vis_path.exists():
        vis_path = checkpoint_dir / "visual_encoder.pt"

    def _load(p):
        d = torch.load(p, map_location=device)
        return d.get("model_state_dict", d.get("model_state", d))

    encoder.load_state_dict(_load(enc_path))
    visual.load_state_dict(_load(vis_path))
    return encoder, visual


def train_mop(mop_model: MoPForecastModel, train_loader,
              args, device) -> None:
    """Train MoP prompts + heads on the LOTSA combined DataLoader.

    The loader returns dict batches with key 'target' of shape (B, full_window, 1).
    We split each window into context (first context_length steps) and target
    (remaining max_horizon steps), then sample a random horizon from HORIZONS.
    """
    train_iter = iter(train_loader)
    max_horizon = max(HORIZONS)

    mop_params = list(mop_model.mop.parameters()) + list(mop_model.heads.parameters())
    optimizer = optim.AdamW(mop_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    for epoch in range(1, args.epochs + 1):
        mop_model.train()
        total_loss, n_valid = 0.0, 0

        for _ in range(args.batches_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # Loader returns dict with 'target': (B, full_window, 1)
            x_full = batch["target"].to(device).float()  # (B, full_window, 1)
            B, full_window, C = x_full.shape

            if full_window < args.context_length + 1:
                continue

            x_in = x_full[:, :args.context_length, :].permute(0, 2, 1)  # (B, C, ctx)
            y_full = x_full[:, args.context_length:, :]                  # (B, max_h, C)

            h_target = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
            if y_full.shape[1] < h_target:
                continue

            y_target = y_full[:, :h_target, :].permute(0, 2, 1)  # (B, C, H)
            pred = mop_model(x_in, h_target)                       # (B*C, H, 1)
            y_ch = y_target.permute(0, 2, 1).reshape(B * C, h_target, 1)
            loss = F.mse_loss(pred, y_ch)

            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(mop_params, 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_valid += 1

        scheduler.step()
        avg = total_loss / max(1, n_valid)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"    Epoch {epoch:3d}/{args.epochs} | Loss: {avg:.4f}")


def eval_mop(mop_model: MoPForecastModel, test_loaders: dict,
             args, device) -> list:
    results = []
    mop_model.eval()
    with torch.no_grad():
        for tag, loader in test_loaders.items():
            for H in HORIZONS:
                all_preds, all_trues = [], []
                for batch in loader:
                    x = batch[0].to(device).float()
                    y = batch[1].to(device).float()
                    if y.shape[1] < H:
                        continue
                    B, L, C = x.shape
                    x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
                    pred = mop_model.greedy_predict(x_in, H, args.context_length)
                    y_target = y[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)
                    all_preds.append(pred)
                    all_trues.append(y_target)
                if not all_preds:
                    continue
                pt  = torch.cat(all_preds)
                tt  = torch.cat(all_trues)
                mse = torch.mean((pt - tt) ** 2).item()
                mae = torch.mean(torch.abs(pt - tt)).item()
                results.append({"dataset": tag, "horizon": H, "mse": mse, "mae": mae})
                print(f"  {tag:12s} H={H:3d}: MSE={mse:.4f}  MAE={mae:.4f}")
    return results


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)

    print("Loading frozen encoders (SimCLR-Full, LOTSA+Chronos)...")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    print(f"  enc_dim={enc_dim}  →  MoP input_dim={enc_dim * 2}")

    max_horizon = max(HORIZONS)

    # Load with context_length = context + max_horizon so we can split x/y internally.
    max_horizon = max(HORIZONS)
    full_window = args.context_length + max_horizon

    # -------------------------------------------------------------------------
    # NORMALISATION WARNING — read before changing norm_mode or the loader.
    #
    # The LOTSA combined loader (build_combined_dataloaders) applies per-series
    # min-max normalisation → values in [0, 1].
    # The ICML test loaders (TimeSeriesDataModule) apply StandardScaler fitted
    # on each dataset's train split → values with mean≈0, std≈1.
    #
    # These two spaces are INCOMPATIBLE: a model trained to predict in [0,1]
    # will produce MSE values ~10–50× higher when evaluated against
    # StandardScaler targets (confirmed empirically: ETT MSE ~2.0 vs ~0.14).
    #
    # The fix used here is norm_mode='revin' in MoPForecastModel:
    #   - RevIN normalises each input instance to mean=0/std=1 at forward time.
    #   - Predictions are denormalised back to the input's own scale.
    #   - This makes the model agnostic to the loader's global scaler.
    #
    # DO NOT use norm_mode='identity' when mixing loaders with different
    # normalisation strategies (e.g., LOTSA min-max + ICML StandardScaler).
    # -------------------------------------------------------------------------

    print("\nBuilding LOTSA+local combined train loader...")
    train_loader, _ = build_combined_dataloaders(
        context_length=full_window,
        batch_size=args.batch_size,
        two_views=False,
        num_workers=args.num_workers,
        pin_memory=True,
        normalize_per_series=True,
    )
    print(f"  Combined loader ready ({len(train_loader.dataset)} series).")

    print("\nBuilding ICML test loaders (zero-shot targets)...")
    test_loaders = build_icml_test_loaders(
        args.icml_data_dir, args.context_length, max(HORIZONS), args.batch_size
    )
    print(f"  {len(test_loaders)} ICML test datasets loaded.")

    summary_rows = []

    for num_prompts in args.num_prompts_values:
        print(f"\n{'='*60}")
        print(f"Ablation: num_prompts={num_prompts}")
        print(f"{'='*60}")

        mop_model = MoPForecastModel(
            encoder=encoder,
            visual_encoder=visual,
            input_dim=enc_dim * 2,
            hidden_dim=args.hidden_dim,
            num_prompts=num_prompts,
            horizons=HORIZONS,
            target_features=1,
            freeze_encoders=True,
            # RevIN normalises per-instance at runtime, making the model
            # agnostic to whether the loader uses min-max or StandardScaler.
            norm_mode="revin",
            scale_cond=False,
        ).to(device)

        t0 = time.time()
        train_mop(mop_model, train_loader, args, device)
        print(f"  Training done in {time.time() - t0:.1f}s")

        # Save checkpoint
        ckpt_path = args.results_dir / f"mop_prompts_{num_prompts}.pt"
        torch.save({"mop_model": mop_model.state_dict(), "num_prompts": num_prompts}, ckpt_path)

        print(f"\n  Zero-shot eval (num_prompts={num_prompts}):")
        results = eval_mop(mop_model, test_loaders, args, device)

        for row in results:
            row["num_prompts"] = num_prompts
        summary_rows.extend(results)

        run_csv = args.results_dir / f"results_prompts_{num_prompts}.csv"
        with open(run_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae", "num_prompts"])
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved → {run_csv}")

    summary_csv = args.results_dir / "ablation_num_prompts_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_prompts", "dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nAll done. Summary → {summary_csv}")


if __name__ == "__main__":
    main()
