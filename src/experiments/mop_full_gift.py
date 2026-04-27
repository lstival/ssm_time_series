"""
Stage 3 — MoMS GIFT-Eval Fine-Tune: fine-tune per-dataset Stage 2 checkpoints
on the GIFT-Eval train split, evaluate on the GIFT-Eval test split.

One MoMS per GIFT-Eval subset, warm-started from the Stage 2 ICML few-shot
checkpoint of the closest related dataset (or the generic Stage 1 checkpoint
if no per-dataset Stage 2 exists).

Input:  --mop_checkpoint_dir   (directory containing Stage 2 per-dataset .pt files)
        --stage1_checkpoint    (fallback Stage 1 .pt, used when no Stage 2 match)
Output: results_dir/mop_gift_{encoder}_results.csv
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
from dataloaders.gift_eval_loader import build_gift_eval_dataloader


class ZeroEncoder(nn.Module):
    """Drop-in visual encoder returning dim-0 tensor — cat([ze, zv]) collapses to ze only."""
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

# GIFT-Eval subsets used for comparison with SEMPO / Chronos
GIFT_SUBSETS = [
    "m_dense_H_long",
    "loop_seattle_H_long",
    "sz_taxi_H_short",
    "solar_H_long",
    "bizitobs_application_10S_long",
    "bizitobs_l2c_H_long",
    "bizitobs_service_10S_long",
    "car_parts_M_short",
    "jena_weather_H_long",
]

# Horizon embedded in each subset name (short=96, long=720)
def _horizon_from_subset(name: str) -> int:
    return 720 if name.endswith("_long") else 96

HORIZONS = [96, 192, 336, 720]


def parse_args():
    p = argparse.ArgumentParser("MoMS Stage 3 — GIFT-Eval Fine-Tune")
    p.add_argument("--encoder_name",        required=True)
    p.add_argument("--checkpoint_dir",      type=Path, required=True,
                   help="Frozen encoder checkpoint directory")
    p.add_argument("--mop_checkpoint_dir",  type=Path, required=True,
                   help="Directory with Stage 2 per-dataset checkpoints")
    p.add_argument("--stage1_checkpoint",   type=Path, required=True,
                   help="Stage 1 checkpoint used as fallback warm-start")
    p.add_argument("--config",              type=Path, required=True)
    p.add_argument("--results_dir",         type=Path,
                   default=root_dir / "results" / "moms_full_pipeline")
    p.add_argument("--few_shot_fraction",   type=float, default=0.10,
                   help="Fraction of GIFT train split for fine-tuning (default 10%)")
    p.add_argument("--finetune_epochs",     type=int,   default=20)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--lr",                  type=float, default=5e-4)
    p.add_argument("--hidden_dim",          type=int,   default=512)
    p.add_argument("--num_prompts",         type=int,   default=16)
    p.add_argument("--context_length",      type=int,   default=336)
    p.add_argument("--num_workers",         type=int,   default=0)
    p.add_argument("--subsets",             nargs="+",  default=GIFT_SUBSETS)
    p.add_argument("--unimodal",            action="store_true",
                   help="Use temporal encoder only (visual branch = zeros)")
    return p.parse_args()


def load_encoders(ckpt_dir: Path, config, device, unimodal: bool = False):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    for name in ["time_series_best.pt", "time_series_encoder.pt"]:
        path = ckpt_dir / name
        if path.exists():
            state = torch.load(path, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            encoder.load_state_dict(state)
            break
    encoder.eval()

    if unimodal:
        visual = ZeroEncoder().to(device)
    else:
        visual = tu.build_visual_encoder_from_config(config.model).to(device)
        for name in ["visual_encoder_best.pt", "visual_encoder.pt"]:
            path = ckpt_dir / name
            if path.exists():
                state = torch.load(path, map_location=device)
                state = state.get("model_state_dict", state.get("model_state", state))
                visual.load_state_dict(state)
                break
        visual.eval()
    return encoder, visual


def _best_stage2_checkpoint(mop_ckpt_dir: Path, encoder_name: str,
                             subset_name: str) -> Optional[Path]:
    """Return the Stage 2 checkpoint that best matches this GIFT subset.

    Matching heuristic (in order):
    1. Exact dataset name fragment in the file name
    2. Frequency match (H→ETT/traffic, M→weather, 10S/T→electricity)
    3. Fallback: None (caller uses Stage 1 checkpoint)
    """
    # Map GIFT frequency tags to ICML dataset tags
    freq_map = {
        "_H_": ["ETTm1", "ETTh1", "ETTm2", "ETTh2", "traffic"],
        "_M_": ["weather", "electricity"],
        "_10S_": ["electricity"],
        "_T_":  ["traffic"],
    }
    candidates = sorted(mop_ckpt_dir.glob(f"mop_fewshot_{encoder_name}_*_checkpoint.pt"))
    if not candidates:
        return None

    # Try keyword match on subset name vs dataset tag
    for cand in candidates:
        tag = cand.stem.replace(f"mop_fewshot_{encoder_name}_", "").replace("_checkpoint", "")
        if tag.lower() in subset_name.lower():
            return cand

    # Try frequency match
    for freq_key, tags in freq_map.items():
        if freq_key in subset_name:
            for cand in candidates:
                for tag in tags:
                    if tag in cand.stem:
                        return cand

    # Return first available as fallback
    return candidates[0]


def build_mop(encoder, visual, enc_dim: int, args, device,
              mop_checkpoint: Optional[Path]) -> MoPForecastModel:
    input_dim = enc_dim if args.unimodal else enc_dim * 2
    mop = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts, horizons=HORIZONS,
        target_features=1, freeze_encoders=True,
        norm_mode="identity",  # z-score done in dataloader; skip internal RevIN
        scale_cond=False,
    ).to(device)
    if mop_checkpoint and mop_checkpoint.exists():
        ckpt = torch.load(mop_checkpoint, map_location=device)
        missing, unexpected = mop.load_state_dict(ckpt["mop_model"], strict=False)
        if unexpected:
            print(f"  [INFO] Ignored checkpoint keys: {unexpected}")
        print(f"  Warm-start: {mop_checkpoint.name}")
    else:
        print("  Warm-start: none (random init)")
    return mop


def finetune(mop: MoPForecastModel, train_loader, args, device):
    params = list(mop.mop.parameters()) + list(mop.heads.parameters())
    opt   = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.finetune_epochs,
                                                  eta_min=args.lr * 0.05)
    for epoch in range(1, args.finetune_epochs + 1):
        mop.train(); total, n = 0.0, 0
        for batch in train_loader:
            # GIFT loader returns dict: {"target": (B,L,1), "future": (B,H,1)}
            x = batch["target"].to(device).float()   # (B, L, 1)
            y = batch["future"].to(device).float()   # (B, H, 1)
            B, L, C = x.shape
            x_in = x.permute(0, 2, 1)  # (B, C, L)
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


def evaluate(mop: MoPForecastModel, test_loader, subset: str, args, device) -> list:
    h_target = _horizon_from_subset(subset)
    eval_horizons = [h for h in HORIZONS if h <= h_target]
    results = []
    mop.eval()
    with torch.no_grad():
        for H in eval_horizons:
            preds, trues = [], []
            for batch in test_loader:
                x = batch["target"].to(device).float()
                y = batch["future"].to(device).float()
                if y.shape[1] < H: continue
                B, L, C = x.shape
                # Direct single-shot prediction — avoids greedy stale RevIN stats bug.
                # Data is already z-scored in the dataloader; norm_mode="identity" skips
                # internal RevIN so predictions stay in normalised space.
                x_in = x.permute(0, 2, 1)  # (B, C, L) — forward expects this shape
                pred = mop(x_in, H)         # (B*C, H, 1)
                y_t  = y[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)
                preds.append(pred); trues.append(y_t)
            if not preds: continue
            pt = torch.cat(preds); tt = torch.cat(trues)
            mse = torch.mean((pt - tt) ** 2).item()
            mae = torch.mean(torch.abs(pt - tt)).item()
            print(f"    {subset:35s} H={H:3d}: MSE={mse:.4f} MAE={mae:.4f}")
            results.append({"encoder": args.encoder_name, "stage": "gift",
                             "subset": subset, "horizon": H, "mse": mse, "mae": mae})
    return results


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)

    print(f"[Stage 3 GIFT-Eval] encoder={args.encoder_name}  mode={'unimodal' if args.unimodal else 'bimodal'}")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device, args.unimodal)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))

    all_results = []
    for subset in args.subsets:
        h_target = _horizon_from_subset(subset)
        print(f"\n--- {subset}  (horizon={h_target}) ---")

        # Build GIFT train/test loaders
        try:
            train_loader = build_gift_eval_dataloader(
                subset_name=subset, context_length=args.context_length,
                prediction_length=h_target, batch_size=args.batch_size,
                num_workers=args.num_workers, force_offline=True,
            )
            test_loader = build_gift_eval_dataloader(
                subset_name=subset, context_length=args.context_length,
                prediction_length=h_target, batch_size=args.batch_size,
                num_workers=args.num_workers, force_offline=True,
            )
        except Exception as e:
            print(f"  [WARN] Could not load {subset}: {e}")
            continue

        # Few-shot subset of train split — fallback to 50% for small datasets
        n_total = len(train_loader.dataset)
        fraction = 0.50 if n_total < 100 else args.few_shot_fraction
        n_few = max(1, int(n_total * fraction))
        few_ds = Subset(train_loader.dataset, list(range(n_few)))
        few_loader = DataLoader(few_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=False)
        print(f"  Few-shot: {n_few}/{n_total} samples (fraction={fraction:.0%})")

        # Warm-start: prefer Stage 2 per-dataset checkpoint, fallback Stage 1
        warm = _best_stage2_checkpoint(args.mop_checkpoint_dir, args.encoder_name, subset)
        if warm is None:
            warm = args.stage1_checkpoint

        mop = build_mop(encoder, visual, enc_dim, args, device, warm)
        finetune(mop, few_loader, args, device)
        results = evaluate(mop, test_loader, subset, args, device)
        all_results.extend(results)
        del mop; torch.cuda.empty_cache()

    csv_path = args.results_dir / f"mop_gift_{args.encoder_name}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder","stage","subset","horizon","mse","mae"])
        writer.writeheader(); writer.writerows(all_results)
    print(f"\nResults → {csv_path}")


if __name__ == "__main__":
    main()
