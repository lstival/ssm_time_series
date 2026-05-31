"""
Stage 2 — MoMS Few-Shot: fine-tune Stage 1 checkpoint on 5% of each ICML dataset,
then evaluate on the test split of that dataset.

One MoMS model per dataset (per-dataset fine-tuning, following SEMPO protocol).
Encoder remains frozen throughout. Only prompts + heads are updated.

Input:  --mop_checkpoint  (Stage 1 output)
Output: results_dir/mop_fewshot_<encoder>_results.csv
        results_dir/mop_fewshot_<encoder>_<dataset>_checkpoint.pt   ← Stage 3 warm-start
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
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


class ZeroEncoder(nn.Module):
    """Drop-in visual encoder returning dim-0 tensor — cat([ze, zv]) collapses to ze only."""
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
]
HORIZONS = [96, 192, 336, 720]


def parse_args():
    p = argparse.ArgumentParser("MoMS Stage 2 — Few-Shot (5% ICML fine-tune)")
    p.add_argument("--encoder_name",       required=True)
    p.add_argument("--checkpoint_dir",     type=Path, required=True,
                   help="Frozen encoder checkpoint directory")
    p.add_argument("--mop_checkpoint",     type=Path, required=True,
                   help="Stage 1 MoMS checkpoint (.pt)")
    p.add_argument("--config",             type=Path, required=True)
    p.add_argument("--icml_data_dir",      type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir",        type=Path, default=root_dir / "results" / "moms_full_pipeline")
    p.add_argument("--few_shot_fraction",  type=float, default=0.05)
    p.add_argument("--finetune_epochs",    type=int,   default=20)
    p.add_argument("--batch_size",         type=int,   default=64)
    p.add_argument("--lr",                 type=float, default=5e-4)
    p.add_argument("--early_stop",         action="store_true",
                   help="Hold out a slice of the few-shot subset and early-stop the "
                        "finetune on it (guards over-specialization, e.g. electricity)")
    p.add_argument("--fewshot_val_frac",   type=float, default=0.2,
                   help="Fraction of the few-shot subset reserved for early-stop validation")
    p.add_argument("--patience",           type=int,   default=3,
                   help="Early-stop patience in epochs (no val improvement)")
    p.add_argument("--hidden_dim",         type=int,   default=512)
    p.add_argument("--num_prompts",        type=int,   default=16)
    p.add_argument("--context_length",     type=int,   default=336)
    p.add_argument("--num_workers",        type=int,   default=0)
    p.add_argument("--unimodal",           action="store_true",
                   help="Use temporal encoder only (visual branch = zeros)")
    p.add_argument("--fusion_mode",        type=str, default="concat",
                   choices=["concat", "film"],
                   help="How z_e and z_v are fused: concat (default) or film (FiLM modulation)")
    p.add_argument("--output_suffix",      type=str, default="",
                   help="Suffix appended to result CSV filename")
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--save_predictions",  action="store_true",
                   help="Save per-dataset predictions/targets to .npz")
    p.add_argument("--predictions_dir",   type=Path, default=None,
                   help="Directory for saved prediction .npz files")
    return p.parse_args()


def resolve_dir(data_dir: Path, ds_name: str) -> str:
    for c in data_dir.rglob(ds_name):
        return str(c.parent)
    return str(data_dir)


def load_encoders(ckpt_dir: Path, config, device, unimodal: bool = False):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    for name in ["time_series_best.pt", "time_series_encoder.pt"]:
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            encoder.load_state_dict(state)
            break
    encoder.eval()

    if unimodal:
        visual = ZeroEncoder().to(device)
    else:
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
    # film fusion: MoP receives enc_dim (z_e only, gated by z_v)
    # concat fusion: MoP receives enc_dim*2
    if args.unimodal:
        input_dim = enc_dim
    elif fusion_mode == "film":
        input_dim = enc_dim
    else:
        input_dim = enc_dim * 2
    mop = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts, horizons=HORIZONS,
        target_features=1, freeze_encoders=True,
        norm_mode="revin",
        scale_cond=False,
        fusion_mode=fusion_mode,
    ).to(device)
    if mop_checkpoint and mop_checkpoint.exists():
        ckpt = torch.load(mop_checkpoint, map_location=device)
        missing, unexpected = mop.load_state_dict(ckpt["mop_model"], strict=False)
        if unexpected:
            print(f"  [INFO] Ignored keys from Stage 1 ckpt (norm_mode mismatch): {unexpected}")
        print(f"  Loaded Stage 1 weights from {mop_checkpoint.name}")
    return mop


_MV_CHUNK = 512  # max B*C per forward pass to avoid OOM on high-channel datasets


@torch.no_grad()
def _val_loss(mop: MoPForecastModel, val_loader, device) -> float:
    """Average MSE over the held-out few-shot slice, across all horizons."""
    mop.eval(); tot, n = 0.0, 0
    for batch in val_loader:
        x = batch[0].to(device).float(); y = batch[1].to(device).float()
        B, L, C = x.shape
        x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
        for h in HORIZONS:
            if y.shape[1] < h: continue
            y_ch = y[:, :h, :].permute(0, 2, 1).reshape(B * C, h, 1)
            for i in range(0, B * C, _MV_CHUNK):
                pred_c = mop(x_in[i:i+_MV_CHUNK], h)
                tot += F.mse_loss(pred_c, y_ch[i:i+_MV_CHUNK]).item(); n += 1
    return tot / max(1, n)


def finetune(mop: MoPForecastModel, train_loader, args, device, val_loader=None):
    params = list(mop.mop.parameters()) + list(mop.heads.parameters())
    opt   = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune_epochs, eta_min=args.lr * 0.05)

    import copy
    best_val = float("inf"); best_state = None; bad = 0
    patience = getattr(args, "patience", 3)

    for epoch in range(1, args.finetune_epochs + 1):
        mop.train(); total, n = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device).float()   # (B, L, C)
            y = batch[1].to(device).float()   # (B, max_h, C)
            B, L, C = x.shape
            x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)  # (B*C, 1, L)
            h = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
            if y.shape[1] < h: continue
            y_ch = y[:, :h, :].permute(0, 2, 1).reshape(B * C, h, 1)
            # chunk B*C to avoid OOM on high-channel datasets
            loss = torch.tensor(0.0, device=device)
            for i in range(0, B * C, _MV_CHUNK):
                pred_c = mop(x_in[i:i+_MV_CHUNK], h)
                loss = loss + F.mse_loss(pred_c, y_ch[i:i+_MV_CHUNK])
            loss = loss / max(1, (B * C + _MV_CHUNK - 1) // _MV_CHUNK)
            if torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total += loss.item(); n += 1
        sched.step()

        # Early stopping on the held-out few-shot slice — guards against
        # over-specialization (electricity over-fit the 5% and lost to zero-shot).
        if val_loader is not None:
            vl = _val_loss(mop, val_loader, device)
            if vl < best_val - 1e-5:
                best_val = vl; best_state = copy.deepcopy(mop.state_dict()); bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"    Early stop at epoch {epoch} (val={best_val:.4f})")
                    break

    if best_state is not None:
        mop.load_state_dict(best_state)
    print(f"    Fine-tune done. Avg loss={total/max(1,n):.4f}"
          + (f"  best_val={best_val:.4f}" if val_loader is not None else ""))


def evaluate(mop: MoPForecastModel, test_loader, tag: str, args, device) -> list:
    results = []
    mop.eval()
    with torch.no_grad():
        pred_store = {}
        for H in HORIZONS:
            preds, trues = [], []
            for batch in test_loader:
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                if y.shape[1] < H: continue
                B, L, C = x.shape
                x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
                y_t  = y[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)
                pred_chunks = []
                for i in range(0, B * C, _MV_CHUNK):
                    pred_chunks.append(
                        mop.greedy_predict(x_in[i:i+_MV_CHUNK], H, args.context_length)
                    )
                pred = torch.cat(pred_chunks, dim=0)
                preds.append(pred); trues.append(y_t)
            if not preds: continue
            pred_cat = torch.cat(preds, dim=0).cpu()
            true_cat = torch.cat(trues, dim=0).cpu()
            mse_vals = []
            mae_vals = []
            for pred, true in zip(preds, trues):
                mse_vals.append(torch.mean((pred - true) ** 2).item())
                mae_vals.append(torch.mean(torch.abs(pred - true)).item())
            mse = sum(mse_vals) / len(mse_vals)
            mae = sum(mae_vals) / len(mae_vals)
            nrmse = mse ** 0.5
            print(f"    {tag:12s} H={H:3d}: MSE={mse:.4f} MAE={mae:.4f} NRMSE={nrmse:.4f}")
            results.append({"encoder": args.encoder_name, "stage": "fewshot",
                             "dataset": tag, "horizon": H, "mse": mse, "mae": mae, "nrmse": nrmse})
            if args.save_predictions:
                pred_store[f"preds_H{H}"] = pred_cat.numpy()
                pred_store[f"targets_H{H}"] = true_cat.numpy()
        if args.save_predictions and pred_store:
            pred_dir = args.predictions_dir or (args.results_dir / "predictions" / "fewshot")
            pred_dir.mkdir(parents=True, exist_ok=True)
            suffix = getattr(args, "output_suffix", "")
            out_path = pred_dir / f"{tag}_fewshot{suffix}.npz"
            np.savez(
                out_path,
                dataset=tag,
                seed=args.seed,
                context_length=args.context_length,
                horizons=np.array(HORIZONS, dtype=np.int32),
                **pred_store,
            )
            print(f"    Saved predictions → {out_path}")
    return results


def main():
    args = parse_args()
    tu.set_seed(args.seed)
    torch.manual_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    max_h  = max(HORIZONS)

    print(f"[Stage 2 Few-Shot] encoder={args.encoder_name}  fraction={args.few_shot_fraction}  mode={'unimodal' if args.unimodal else 'bimodal'}")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device, args.unimodal)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))

    all_results = []
    for ds_file in ICML_DATASETS:
        tag      = ds_file.replace(".csv", "")
        ds_dir   = resolve_dir(args.icml_data_dir, ds_file)
        print(f"\n--- {tag} ---")

        try:
            mod_train = TimeSeriesDataModule(
                dataset_name=ds_file, data_dir=ds_dir,
                batch_size=args.batch_size, val_batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=False, normalize=True,
                train=True, val=False, test=False,
                sample_size=(args.context_length, 0, max_h), scaler_type="standard",
                features='M',
            )
            mod_train.setup()
            full_train_loader = mod_train.train_loaders[0] if mod_train.train_loaders else None

            mod_test = TimeSeriesDataModule(
                dataset_name=ds_file, data_dir=ds_dir,
                batch_size=args.batch_size, val_batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=False, normalize=True,
                train=False, val=False, test=True,
                sample_size=(args.context_length, 0, max_h), scaler_type="standard",
                features='M',
            )
            mod_test.setup()
            test_loader = mod_test.test_loaders[0] if mod_test.test_loaders else None
        except Exception as e:
            print(f"  [WARN] {tag}: {e}")
            continue

        if full_train_loader is None or test_loader is None:
            print(f"  [WARN] {tag}: missing loader, skipping")
            continue

        # Few-shot subset: sample the fraction RANDOMLY across the whole train
        # period. The old `range(n_few)` took a contiguous slice from the very
        # start of training — for non-stationary series (e.g. electricity) that
        # slice is unrepresentative of the test tail and degrades vs zero-shot
        # (diagnosed: first=+26% MSE, random≈zero-shot). Random is the safe
        # universal choice.
        full_ds  = full_train_loader.dataset
        N        = len(full_ds)
        n_few    = max(1, int(N * args.few_shot_fraction))
        rng      = np.random.default_rng(args.seed)
        sel      = rng.choice(N, size=n_few, replace=False)
        sel.sort()
        # Hold out a slice of the few-shot subset for early-stopping so the
        # finetune doesn't over-specialize (electricity over-fit the 5%).
        n_val    = max(1, int(n_few * args.fewshot_val_frac)) if args.early_stop else 0
        val_idx  = sel[:n_val].tolist()
        train_idx = sel[n_val:].tolist()
        few_ds   = Subset(full_ds, train_idx)
        few_loader = DataLoader(few_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=False)
        val_loader = None
        if n_val > 0:
            val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers, drop_last=False)
        print(f"  Few-shot: {len(train_idx)}/{N} train"
              + (f" + {n_val} val (early-stop)" if n_val else "") + " samples (random)")

        # Fresh MoP per dataset, warm-started from Stage 1
        mop = build_mop(encoder, visual, enc_dim, args, device, args.mop_checkpoint)
        finetune(mop, few_loader, args, device, val_loader=val_loader)

        # Save per-dataset checkpoint for Stage 3
        ds_ckpt = args.results_dir / f"mop_fewshot_{args.encoder_name}_{tag}_checkpoint.pt"
        torch.save({
            "mop_model": mop.state_dict(),
            "encoder_name": args.encoder_name,
            "dataset": tag,
            "enc_dim": enc_dim,
            "args": args,
        }, ds_ckpt)
        print(f"  Checkpoint → {ds_ckpt.name}")

        results = evaluate(mop, test_loader, tag, args, device)
        all_results.extend(results)
        del mop; torch.cuda.empty_cache()

    suffix = getattr(args, "output_suffix", "")
    csv_path = args.results_dir / f"mop_fewshot_{args.encoder_name}{suffix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder","stage","dataset","horizon","mse","mae","nrmse"])
        writer.writeheader(); writer.writerows(all_results)
    print(f"\nResults → {csv_path}")


if __name__ == "__main__":
    main()
