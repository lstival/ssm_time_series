"""
MoP Tuning — ICML Leave-One-Out (Exp B).

For each of the 7 ICML benchmarks:
  - Tune MoP + heads on the train splits of the OTHER 6 datasets
  - Evaluate zero-shot on the test split of the held-out dataset

This is a true cross-dataset zero-shot: the model never sees the target
domain during MoP tuning. The encoder remains fully frozen throughout.
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

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
]
HORIZONS = [96, 192, 336, 720]


def parse_args():
    p = argparse.ArgumentParser("MoP Tuning — ICML Leave-One-Out (Exp B)")
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=src_dir / "configs" / "lotsa_simclr_bimodal_nano.yaml")
    p.add_argument("--data_dir", type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir", type=Path, default=root_dir / "results" / "mop_tuning")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_prompts", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--context_length", type=int, default=336)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--norm_mode", type=str, default="identity",
                   choices=["revin", "minmax", "identity"],
                   help="MoP internal norm. 'identity' avoids double-norm with StandardScaler loader.")
    p.add_argument("--scale_cond", action="store_true", default=False,
                   help="Condition MoP prompts on RevIN stats (only active when norm_mode=revin).")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Optional suffix for the output CSV filename, e.g. '_revin'.")
    return p.parse_args()


def resolve_dir(data_dir, ds_name):
    for c in data_dir.rglob(ds_name):
        return str(c.parent)
    return str(data_dir)


def build_loaders(data_dir, ds_names, context_length, max_horizon, batch_size, train=True, test=False):
    loaders = {}
    for ds_name in ds_names:
        tag = ds_name.replace(".csv", "")
        resolved = resolve_dir(data_dir, ds_name)
        try:
            module = TimeSeriesDataModule(
                dataset_name=ds_name,
                data_dir=resolved,
                batch_size=batch_size,
                val_batch_size=batch_size,
                num_workers=0,
                pin_memory=False,
                normalize=True,
                train=train, val=False, test=test,
                sample_size=(context_length, 0, max_horizon),
                scaler_type="standard",
            )
            module.setup()
            if train and module.train_loaders:
                loaders[tag] = module.train_loaders[0]
            elif test and module.test_loaders:
                loaders[tag] = module.test_loaders[0]
        except Exception as e:
            print(f"  Skipping {tag}: {e}")
    return loaders


def build_fresh_mop(encoder, visual, enc_dim, args, device):
    # scale_cond requires RevIN stats; disable it when norm_mode='identity'
    scale_cond = args.scale_cond and (args.norm_mode == "revin")
    return MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=enc_dim * 2, hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts, horizons=HORIZONS,
        target_features=1, freeze_encoders=True,
        norm_mode=args.norm_mode,
        scale_cond=scale_cond,
    ).to(device)


def load_encoders(checkpoint_dir, config, device):
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


def tune_and_eval(mop_model, train_loaders, test_loader, test_tag, args, device):
    loader_pool = list(train_loaders.items())
    iters = {tag: iter(loader) for tag, loader in train_loaders.items()}

    mop_params = list(mop_model.mop.parameters()) + list(mop_model.heads.parameters())
    optimizer = optim.AdamW(mop_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    max_horizon = max(HORIZONS)

    for epoch in range(1, args.epochs + 1):
        mop_model.train()
        total_loss, n_valid = 0.0, 0
        for step in range(args.batches_per_epoch):
            tag, loader = loader_pool[step % len(loader_pool)]
            try:
                batch = next(iters[tag])
            except StopIteration:
                iters[tag] = iter(train_loaders[tag])
                batch = next(iters[tag])

            x = batch[0].to(device).float()
            y = batch[1].to(device).float()
            B, L, C = x.shape
            x_in = x.permute(0, 2, 1)
            y_in = y.permute(0, 2, 1)

            h_target = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
            if h_target > y_in.shape[2]:
                continue

            y_target = y_in[:, :, :h_target]
            pred = mop_model(x_in, h_target)
            y_target_ch = y_target.permute(0, 2, 1).reshape(B * C, h_target, 1)
            loss = F.mse_loss(pred, y_target_ch)

            if torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(mop_params, 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_valid += 1

        scheduler.step()
        avg = total_loss / max(1, n_valid)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"    Epoch {epoch:2d}/{args.epochs} | Loss: {avg:.4f}")

    # Zero-shot eval on held-out dataset
    results = []
    mop_model.eval()
    with torch.no_grad():
        for H in HORIZONS:
            all_preds, all_trues = [], []
            for batch in test_loader:
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
            print(f"  {test_tag} H={H}: MSE={mse:.4f} MAE={mae:.4f}")
            results.append({"dataset": test_tag, "horizon": H, "mse": mse, "mae": mae})
    return results


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)

    print("Loading encoders (frozen)...")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))

    max_horizon = max(HORIZONS)

    all_results = []

    for held_out in ICML_DATASETS:
        held_tag = held_out.replace(".csv", "")
        train_names = [d for d in ICML_DATASETS if d != held_out]
        print(f"\n{'='*60}")
        print(f"LOO fold: held-out={held_tag} | training on {[d.replace('.csv','') for d in train_names]}")
        print(f"{'='*60}")

        train_loaders = build_loaders(args.data_dir, train_names, args.context_length, max_horizon, args.batch_size, train=True)
        test_loaders  = build_loaders(args.data_dir, [held_out],  args.context_length, max_horizon, args.batch_size, test=True)

        if held_tag not in test_loaders or not train_loaders:
            print(f"  Skipping {held_tag}: missing loaders.")
            continue

        # Fresh MoP for each fold (encoder weights are shared/frozen)
        mop_model = build_fresh_mop(encoder, visual, enc_dim, args, device)

        fold_results = tune_and_eval(
            mop_model, train_loaders, test_loaders[held_tag], held_tag, args, device
        )
        all_results.extend(fold_results)

    out_csv = args.results_dir / f"mop_zeroshot_loo_results{args.output_suffix}.csv"
    with open(out_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nDone. Results saved to {out_csv}")


if __name__ == "__main__":
    main()
