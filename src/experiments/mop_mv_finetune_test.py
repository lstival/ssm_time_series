"""
Fine-tuning test on multivariate-heavy datasets (weather, traffic, electricity).

Goal: check if per-dataset fine-tuning with more data improves MoMS on the 3
worst-performing datasets, to validate whether adding similar multivariate
datasets to the Stage 1 training corpus would help.

Tests fractions: 5%, 20%, 50%, 100% of training data.
Uses CLIP-Full MoMS Stage 1 checkpoint as warm-start.

Output: results_dir/mop_mv_finetune_<encoder>_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent
root_dir   = src_dir.parent
for p in (src_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from time_series_loader import TimeSeriesDataModule

MV_DATASETS = ["weather.csv", "traffic.csv", "electricity.csv"]
HORIZONS    = [96, 192, 336, 720]
TEST_FRACTIONS = [0.05, 0.20, 0.50, 1.00]


def parse_args():
    p = argparse.ArgumentParser("MoMS MV Fine-Tune Test")
    p.add_argument("--encoder_name",      required=True)
    p.add_argument("--checkpoint_dir",    type=Path, required=True)
    p.add_argument("--mop_checkpoint",    type=Path, required=True,
                   help="Stage 1 MoMS checkpoint (.pt)")
    p.add_argument("--config",            type=Path, required=True)
    p.add_argument("--icml_data_dir",     type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir",       type=Path, default=root_dir / "results" / "mop_mv_finetune")
    p.add_argument("--fractions",         type=float, nargs="+", default=TEST_FRACTIONS)
    p.add_argument("--finetune_epochs",   type=int,   default=20)
    p.add_argument("--batch_size",        type=int,   default=64)
    p.add_argument("--lr",                type=float, default=5e-4)
    p.add_argument("--hidden_dim",        type=int,   default=512)
    p.add_argument("--num_prompts",       type=int,   default=16)
    p.add_argument("--context_length",    type=int,   default=336)
    p.add_argument("--num_workers",       type=int,   default=0)
    p.add_argument("--fusion_mode",       type=str,   default="concat",
                   choices=["concat", "film"])
    return p.parse_args()


def resolve_dir(data_dir: Path, ds_name: str) -> str:
    for c in data_dir.rglob(ds_name):
        return str(c.parent)
    return str(data_dir)


def load_encoders(ckpt_dir: Path, config, device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    for name in ["time_series_best.pt", "time_series_encoder.pt"]:
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            encoder.load_state_dict(state)
            break
    encoder.eval()

    visual = tu.build_visual_encoder_from_config(config.model).to(device)
    for name in ["visual_encoder_best.pt", "visual_encoder.pt"]:
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            visual.load_state_dict(state)
            break
    visual.eval()
    return encoder, visual


def build_mop(encoder, visual, enc_dim: int, args, device,
              mop_checkpoint: Optional[Path] = None) -> MoPForecastModel:
    fusion_mode = getattr(args, "fusion_mode", "concat")
    input_dim = enc_dim if fusion_mode == "film" else enc_dim * 2
    mop = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts, horizons=HORIZONS,
        target_features=1, freeze_encoders=True,
        norm_mode="revin", scale_cond=False,
        fusion_mode=fusion_mode,
    ).to(device)
    if mop_checkpoint and mop_checkpoint.exists():
        ckpt = torch.load(mop_checkpoint, map_location=device)
        missing, unexpected = mop.load_state_dict(ckpt["mop_model"], strict=False)
        if unexpected:
            print(f"  [INFO] Ignored keys: {unexpected}")
        print(f"  Loaded Stage 1 weights from {mop_checkpoint.name}")
    return mop


def finetune(mop: MoPForecastModel, train_loader, args, device):
    params = list(mop.mop.parameters()) + list(mop.heads.parameters())
    opt   = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune_epochs,
                                                  eta_min=args.lr * 0.05)
    for epoch in range(1, args.finetune_epochs + 1):
        mop.train(); total, n = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device).float()
            y = batch[1].to(device).float()
            B, L, C = x.shape
            x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
            h = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
            if y.shape[1] < h: continue
            y_ch = y[:, :h, :].permute(0, 2, 1).reshape(B * C, h, 1)
            pred = mop(x_in, h)
            loss = F.mse_loss(pred, y_ch)
            if torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total += loss.item(); n += 1
        sched.step()
    print(f"    Fine-tune done. Avg loss={total/max(1,n):.4f}")


def evaluate(mop: MoPForecastModel, test_loader, tag: str, args, device) -> list:
    results = []
    mop.eval()
    with torch.no_grad():
        for H in HORIZONS:
            preds, trues = [], []
            for batch in test_loader:
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                if y.shape[1] < H: continue
                B, L, C = x.shape
                x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
                pred = mop.greedy_predict(x_in, H, args.context_length)
                y_t  = y[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)
                preds.append(pred); trues.append(y_t)
            if not preds: continue
            pt = torch.cat(preds); tt = torch.cat(trues)
            mse = torch.mean((pt - tt) ** 2).item()
            mae = torch.mean(torch.abs(pt - tt)).item()
            print(f"      H={H:3d}: MSE={mse:.4f} MAE={mae:.4f}")
            results.append({"encoder": args.encoder_name, "dataset": tag,
                             "horizon": H, "mse": mse, "mae": mae})
    return results


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    max_h  = max(HORIZONS)

    print(f"[MV Fine-Tune Test] encoder={args.encoder_name}  fractions={args.fractions}")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))

    all_results = []

    for ds_file in MV_DATASETS:
        tag    = ds_file.replace(".csv", "")
        ds_dir = resolve_dir(args.icml_data_dir, ds_file)
        print(f"\n{'='*50}")
        print(f"Dataset: {tag}")

        try:
            mod_train = TimeSeriesDataModule(
                dataset_name=ds_file, data_dir=ds_dir,
                batch_size=args.batch_size, val_batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=False, normalize=True,
                train=True, val=False, test=False,
                sample_size=(args.context_length, 0, max_h), scaler_type="standard",
            )
            mod_train.setup()
            full_train_loader = mod_train.train_loaders[0] if mod_train.train_loaders else None

            mod_test = TimeSeriesDataModule(
                dataset_name=ds_file, data_dir=ds_dir,
                batch_size=args.batch_size, val_batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=False, normalize=True,
                train=False, val=False, test=True,
                sample_size=(args.context_length, 0, max_h), scaler_type="standard",
            )
            mod_test.setup()
            test_loader = mod_test.test_loaders[0] if mod_test.test_loaders else None
        except Exception as e:
            print(f"  [WARN] {tag}: {e}")
            continue

        if not full_train_loader or not test_loader:
            print(f"  [WARN] {tag}: missing loader")
            continue

        full_ds = full_train_loader.dataset
        print(f"  Train pool: {len(full_ds):,} samples | Test: {len(test_loader.dataset):,}")

        # Zero-shot baseline (no fine-tuning)
        print(f"\n  [fraction=0.00 — zero-shot baseline]")
        mop = build_mop(encoder, visual, enc_dim, args, device, args.mop_checkpoint)
        zs_results = evaluate(mop, test_loader, tag, args, device)
        for r in zs_results:
            r["fraction"] = 0.0
        all_results.extend(zs_results)
        del mop; torch.cuda.empty_cache()

        # Fine-tuning at each fraction
        for frac in args.fractions:
            n_few = max(1, int(len(full_ds) * frac))
            print(f"\n  [fraction={frac:.2f} — {n_few:,}/{len(full_ds):,} samples]")
            few_loader = DataLoader(
                Subset(full_ds, list(range(n_few))),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, drop_last=False
            )
            # Always warm-start from Stage 1 (not from previous fraction)
            mop = build_mop(encoder, visual, enc_dim, args, device, args.mop_checkpoint)
            finetune(mop, few_loader, args, device)
            results = evaluate(mop, test_loader, tag, args, device)
            for r in results:
                r["fraction"] = frac
            all_results.extend(results)
            del mop; torch.cuda.empty_cache()

    # Save results
    csv_path = args.results_dir / f"mop_mv_finetune_{args.encoder_name}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder","dataset","fraction","horizon","mse","mae"])
        writer.writeheader(); writer.writerows(all_results)
    print(f"\nResults → {csv_path}")

    # Summary table
    from collections import defaultdict
    print("\n=== Summary: avg MSE per dataset × fraction ===")
    summary = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        summary[r["dataset"]][r["fraction"]].append(r["mse"])

    fracs = sorted({r["fraction"] for r in all_results})
    header = f"{'Dataset':<14}" + "".join(f" {f:>8.0%}" for f in fracs)
    print(header)
    print("-" * len(header))
    for ds in [d.replace(".csv","") for d in MV_DATASETS]:
        row = f"{ds:<14}"
        for frac in fracs:
            vals = summary[ds][frac]
            row += f" {sum(vals)/len(vals):>8.3f}" if vals else f" {'—':>8}"
        print(row)


if __name__ == "__main__":
    main()
