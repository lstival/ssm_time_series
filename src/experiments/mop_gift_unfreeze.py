"""
Stage 3 variant — MoMS GIFT-Eval with gradual encoder unfreezing.

Goal: test whether fine-tuning the encoder on GIFT domain data improves
performance vs frozen encoder (upper-bound check for encoder capacity).

Unfreezing schedule (per subset):
  Phase 1 (epochs 1–phase1_epochs):   heads + MoP only, encoder frozen, lr=head_lr
  Phase 2 (epochs phase1+1–phase2):   + last encoder block unfrozen, lr=enc_lr_last
  Phase 3 (epochs phase2+1–total):    + all encoder blocks unfrozen, lr=enc_lr_full

100% of GIFT train data used (supervised upper bound).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, List

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

HORIZONS = [96, 192, 336, 720]


def _horizon_from_subset(name: str) -> int:
    return 720 if name.endswith("_long") else 96


def parse_args():
    p = argparse.ArgumentParser("MoMS GIFT — Gradual Encoder Unfreeze")
    p.add_argument("--encoder_name",       required=True)
    p.add_argument("--checkpoint_dir",     type=Path, required=True)
    p.add_argument("--mop_checkpoint_dir", type=Path, required=True)
    p.add_argument("--stage1_checkpoint",  type=Path, required=True)
    p.add_argument("--config",             type=Path, required=True)
    p.add_argument("--results_dir",        type=Path,
                   default=root_dir / "results" / "moms_gift_unfreeze")
    p.add_argument("--few_shot_fraction",  type=float, default=1.0)
    p.add_argument("--phase1_epochs",      type=int,   default=10,
                   help="Epochs with encoder fully frozen (heads + MoP only)")
    p.add_argument("--phase2_epochs",      type=int,   default=10,
                   help="Epochs with last encoder block unfrozen")
    p.add_argument("--phase3_epochs",      type=int,   default=30,
                   help="Epochs with full encoder unfrozen")
    p.add_argument("--head_lr",            type=float, default=5e-4)
    p.add_argument("--enc_lr_last",        type=float, default=1e-4)
    p.add_argument("--enc_lr_full",        type=float, default=5e-5)
    p.add_argument("--batch_size",         type=int,   default=64)
    p.add_argument("--hidden_dim",         type=int,   default=512)
    p.add_argument("--num_prompts",        type=int,   default=16)
    p.add_argument("--context_length",     type=int,   default=336)
    p.add_argument("--num_workers",        type=int,   default=0)
    p.add_argument("--subsets",            nargs="+",  default=GIFT_SUBSETS)
    return p.parse_args()


def load_encoders(ckpt_dir: Path, config, device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    for name in ["time_series_best.pt", "time_series_encoder.pt"]:
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            encoder.load_state_dict(state)
            break
    visual = tu.build_visual_encoder_from_config(config.model).to(device)
    for name in ["visual_encoder_best.pt", "visual_encoder.pt"]:
        p = ckpt_dir / name
        if p.exists():
            state = torch.load(p, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            visual.load_state_dict(state)
            break
    return encoder, visual


def _last_block_params(encoder: nn.Module) -> List[nn.Parameter]:
    """Return parameters of the last block only (for phase 2 unfreeze)."""
    blocks = None
    if hasattr(encoder, 'blocks'):
        blocks = list(encoder.blocks)
    elif hasattr(encoder, 'layers'):
        blocks = list(encoder.layers)
    if blocks:
        return list(blocks[-1].parameters())
    # Fallback: return output projection if no blocks
    for name in ['output_proj', 'out_proj', 'fc']:
        if hasattr(encoder, name):
            return list(getattr(encoder, name).parameters())
    return []


def build_mop(encoder, visual, enc_dim: int, args, device,
              mop_checkpoint: Optional[Path]) -> MoPForecastModel:
    mop = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=enc_dim * 2, hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts, horizons=HORIZONS,
        target_features=1, freeze_encoders=True,  # start frozen, unfreeze manually
        norm_mode="identity",
        scale_cond=False,
    ).to(device)
    if mop_checkpoint and mop_checkpoint.exists():
        ckpt = torch.load(mop_checkpoint, map_location=device)
        missing, unexpected = mop.load_state_dict(ckpt["mop_model"], strict=False)
        if unexpected:
            print(f"  [INFO] Ignored keys: {unexpected}")
        print(f"  Warm-start: {mop_checkpoint.name}")
    else:
        print("  Warm-start: none (random init)")
    return mop


def _best_stage2_checkpoint(mop_ckpt_dir: Path, encoder_name: str,
                             subset_name: str) -> Optional[Path]:
    freq_map = {
        "_H_": ["ETTm1", "ETTh1", "ETTm2", "ETTh2", "traffic"],
        "_M_": ["weather", "electricity"],
        "_10S_": ["electricity"],
        "_T_":  ["traffic"],
    }
    candidates = sorted(mop_ckpt_dir.glob(f"mop_fewshot_{encoder_name}_*_checkpoint.pt"))
    if not candidates:
        return None
    for cand in candidates:
        tag = cand.stem.replace(f"mop_fewshot_{encoder_name}_", "").replace("_checkpoint", "")
        if tag.lower() in subset_name.lower():
            return cand
    for freq_key, tags in freq_map.items():
        if freq_key in subset_name:
            for cand in candidates:
                for tag in tags:
                    if tag in cand.stem:
                        return cand
    return candidates[0]


def finetune_gradual(mop: MoPForecastModel, train_loader, args, device):
    """Three-phase gradual unfreeze training."""
    phases = [
        (args.phase1_epochs, "heads+MoP only",       None,             args.head_lr),
        (args.phase2_epochs, "last encoder block",   "last_block",     args.enc_lr_last),
        (args.phase3_epochs, "full encoder",          "full",           args.enc_lr_full),
    ]

    for phase_epochs, phase_name, unfreeze_mode, lr in phases:
        if phase_epochs <= 0:
            continue

        # --- unfreeze ---
        if unfreeze_mode == "last_block":
            last_params = _last_block_params(mop.encoder)
            for p in last_params:
                p.requires_grad = True
            mop.encoder.train()
            print(f"\n  [Phase: {phase_name}] unfreezing {len(last_params)} params, lr={lr}")
        elif unfreeze_mode == "full":
            for p in mop.encoder.parameters():
                p.requires_grad = True
            for p in mop.visual_encoder.parameters():
                p.requires_grad = True
            mop.encoder.train()
            mop.visual_encoder.train()
            print(f"\n  [Phase: {phase_name}] full encoder unfrozen, lr={lr}")
        else:
            print(f"\n  [Phase: {phase_name}] encoder frozen, lr={lr}")

        trainable = [p for p in mop.parameters() if p.requires_grad]
        opt   = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=phase_epochs, eta_min=lr * 0.05)

        for epoch in range(1, phase_epochs + 1):
            mop.train()
            # keep frozen encoders in eval mode so BN/dropout stay stable
            if unfreeze_mode is None:
                mop.encoder.eval()
                mop.visual_encoder.eval()
            total, n = 0.0, 0
            for batch in train_loader:
                x = batch["target"].to(device).float()
                y = batch["future"].to(device).float()
                B, L, C = x.shape
                x_in = x.permute(0, 2, 1)  # (B, C, L)
                h = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
                if y.shape[1] < h:
                    continue
                y_ch = y[:, :h, :].permute(0, 2, 1).reshape(B * C, h, 1)
                pred = mop(x_in, h)
                loss = F.mse_loss(pred, y_ch)
                if torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                    total += loss.item(); n += 1
            sched.step()
            if epoch % 5 == 0 or epoch == phase_epochs:
                print(f"    epoch {epoch:3d}/{phase_epochs}  loss={total/max(1,n):.4f}")


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
                if y.shape[1] < H:
                    continue
                B, L, C = x.shape
                x_in = x.permute(0, 2, 1)
                pred = mop(x_in, H)
                y_t  = y[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)
                preds.append(pred); trues.append(y_t)
            if not preds:
                continue
            pt = torch.cat(preds); tt = torch.cat(trues)
            mse = torch.mean((pt - tt) ** 2).item()
            mae = torch.mean(torch.abs(pt - tt)).item()
            print(f"    {subset:35s} H={H:3d}: MSE={mse:.4f} MAE={mae:.4f}")
            results.append({"encoder": args.encoder_name, "stage": "gift_unfreeze",
                             "subset": subset, "horizon": H, "mse": mse, "mae": mae})
    return results


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)

    total_epochs = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs
    print(f"[GIFT Gradual Unfreeze] encoder={args.encoder_name}")
    print(f"  Phase1={args.phase1_epochs}ep lr={args.head_lr} | "
          f"Phase2={args.phase2_epochs}ep lr={args.enc_lr_last} | "
          f"Phase3={args.phase3_epochs}ep lr={args.enc_lr_full} | "
          f"total={total_epochs}ep")

    encoder, visual = load_encoders(args.checkpoint_dir, config, device)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))

    all_results = []
    for subset in args.subsets:
        h_target = _horizon_from_subset(subset)
        print(f"\n{'='*55}")
        print(f"Subset: {subset}  (H={h_target})")

        try:
            train_hf = build_gift_eval_dataloader(
                subset_name=subset, context_length=args.context_length,
                prediction_length=h_target, batch_size=args.batch_size,
                num_workers=args.num_workers, force_offline=True,
            )
            test_hf = build_gift_eval_dataloader(
                subset_name=subset, context_length=args.context_length,
                prediction_length=h_target, batch_size=args.batch_size,
                num_workers=args.num_workers, force_offline=True,
            )
        except Exception as e:
            print(f"  [WARN] Could not load {subset}: {e}")
            continue

        n_total = len(train_hf.dataset)
        fraction = 0.50 if n_total < 100 else args.few_shot_fraction
        n_few = max(1, int(n_total * fraction))
        few_ds = Subset(train_hf.dataset, list(range(n_few)))
        few_loader = DataLoader(few_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=False)
        print(f"  Train: {n_few}/{n_total} samples (fraction={fraction:.0%})")

        warm = _best_stage2_checkpoint(args.mop_checkpoint_dir, args.encoder_name, subset)
        if warm is None:
            warm = args.stage1_checkpoint

        # Re-load fresh encoder for each subset (avoid cross-subset encoder drift)
        encoder, visual = load_encoders(args.checkpoint_dir, config, device)
        mop = build_mop(encoder, visual, enc_dim, args, device, warm)

        finetune_gradual(mop, few_loader, args, device)
        results = evaluate(mop, test_hf, subset, args, device)
        all_results.extend(results)
        del mop; torch.cuda.empty_cache()

    csv_path = args.results_dir / f"mop_gift_unfreeze_{args.encoder_name}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder","stage","subset","horizon","mse","mae"])
        writer.writeheader(); writer.writerows(all_results)
    print(f"\nResults → {csv_path}")

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    if not df.empty:
        avg = df.groupby("subset")["mse"].mean().sort_values()
        print("\n=== avg MSE per subset ===")
        for s, v in avg.items():
            print(f"  {s:<40} {v:.4f}")
        print(f"\nOverall avg MSE: {df['mse'].mean():.4f}")


if __name__ == "__main__":
    main()
