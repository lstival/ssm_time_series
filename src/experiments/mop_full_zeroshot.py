"""
Stage 1 — MoMS Zero-Shot: train on LOTSA+Chronos, eval on 7 ICML benchmarks.

The encoder is fully frozen. Only MoMS prompts + heads are trained.
norm_mode='revin' so the model is agnostic to LOTSA min-max vs ICML StandardScaler.

Saves:
    results_dir/mop_zeroshot_<encoder>_results.csv
    results_dir/mop_zeroshot_<encoder>_checkpoint.pt   ← used as Stage 2 warm-start
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
src_dir    = script_dir.parent
root_dir   = src_dir.parent
for p in (src_dir, root_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from time_series_loader import TimeSeriesDataModule
from dataloaders.local_dataset_loader import build_combined_dataloaders


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
    p = argparse.ArgumentParser("MoMS Stage 1 — Zero-Shot (LOTSA+Chronos → ICML)")
    p.add_argument("--encoder_name",    required=True,
                   help="Short label used in output filenames, e.g. simclr, byol, clip, gram")
    p.add_argument("--checkpoint_dir",  type=Path, required=True)
    p.add_argument("--config",          type=Path, required=True)
    p.add_argument("--icml_data_dir",   type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir",     type=Path, default=root_dir / "results" / "moms_full_pipeline")
    p.add_argument("--epochs",          type=int,  default=50)
    p.add_argument("--batch_size",      type=int,  default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--hidden_dim",      type=int,  default=512)
    p.add_argument("--num_prompts",     type=int,  default=16)
    p.add_argument("--context_length",  type=int,  default=336)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--num_workers",     type=int,  default=4)
    p.add_argument("--unimodal",        action="store_true",
                   help="Use temporal encoder only (visual branch = zeros)")
    p.add_argument("--fusion_mode",     type=str, default="concat",
                   choices=["concat", "film"],
                   help="How z_e and z_v are fused: concat (default) or film")
    p.add_argument("--output_suffix",   type=str, default="",
                   help="Suffix appended to result/checkpoint filenames")
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


def build_icml_test_loaders(icml_dir: Path, ctx: int, max_h: int, bs: int) -> dict:
    loaders = {}
    for ds in ICML_DATASETS:
        tag = ds.replace(".csv", "")
        try:
            mod = TimeSeriesDataModule(
                dataset_name=ds, data_dir=resolve_dir(icml_dir, ds),
                batch_size=bs, val_batch_size=bs, num_workers=0,
                pin_memory=False, normalize=True,
                train=False, val=False, test=True,
                sample_size=(ctx, 0, max_h), scaler_type="standard",
            )
            mod.setup()
            if mod.test_loaders:
                loaders[tag] = mod.test_loaders[0]
        except Exception as e:
            print(f"  [WARN] {tag}: {e}")
    return loaders


def train(mop: MoPForecastModel, train_loader, args, device):
    max_h = max(HORIZONS)
    full_window = args.context_length + max_h
    params = list(mop.mop.parameters()) + list(mop.heads.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    it = iter(train_loader)

    for epoch in range(1, args.epochs + 1):
        mop.train(); total, n = 0.0, 0
        for _ in range(args.batches_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader); batch = next(it)

            x_full = batch["target"].to(device).float()   # (B, full_window, 1)
            B, W, C = x_full.shape
            if W < args.context_length + 1:
                continue

            x_in = x_full[:, :args.context_length, :].permute(0, 2, 1)   # (B,C,L)
            y_full = x_full[:, args.context_length:, :]                    # (B,max_h,1)
            h = HORIZONS[torch.randint(0, len(HORIZONS), (1,)).item()]
            if y_full.shape[1] < h:
                continue

            y_ch = y_full[:, :h, :].permute(0, 2, 1).reshape(B * C, h, 1)
            pred = mop(x_in, h)
            loss = F.mse_loss(pred, y_ch)
            if torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total += loss.item(); n += 1

        sched.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} loss={total/max(1,n):.4f}")


def evaluate(mop: MoPForecastModel, test_loaders: dict, args, device) -> list:
    results = []
    mop.eval()
    with torch.no_grad():
        for tag, loader in test_loaders.items():
            for H in HORIZONS:
                preds, trues = [], []
                for batch in loader:
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
                print(f"  {tag:12s} H={H:3d}: MSE={mse:.4f} MAE={mae:.4f}")
                results.append({"dataset": tag, "horizon": H, "mse": mse, "mae": mae,
                                 "encoder": args.encoder_name, "stage": "zeroshot"})
    return results


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)

    mode = "unimodal" if args.unimodal else "bimodal"
    print(f"[Stage 1 Zero-Shot] encoder={args.encoder_name}  mode={mode}")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device, args.unimodal)
    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    fusion_mode = getattr(args, "fusion_mode", "concat")
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
        norm_mode="revin", scale_cond=False,
        fusion_mode=fusion_mode,
    ).to(device)

    full_window = args.context_length + max(HORIZONS)
    print("Building LOTSA+local train loader...")
    train_loader, _ = build_combined_dataloaders(
        context_length=full_window, batch_size=args.batch_size,
        two_views=False, num_workers=args.num_workers,
        pin_memory=True, normalize_per_series=True,
    )
    print(f"  {len(train_loader.dataset):,} series in combined corpus")

    print("Training MoMS prompts + heads...")
    t0 = time.time()
    train(mop, train_loader, args, device)
    print(f"  Done in {time.time()-t0:.0f}s")

    suffix = getattr(args, "output_suffix", "")
    ckpt_path = args.results_dir / f"mop_zeroshot_{args.encoder_name}{suffix}_checkpoint.pt"
    torch.save({
        "mop_model": mop.state_dict(),
        "encoder_name": args.encoder_name,
        "enc_dim": enc_dim,
        "args": args,
    }, ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    print("\nZero-shot evaluation on ICML test splits...")
    test_loaders = build_icml_test_loaders(
        args.icml_data_dir, args.context_length, max(HORIZONS), args.batch_size)
    results = evaluate(mop, test_loaders, args, device)

    csv_path = args.results_dir / f"mop_zeroshot_{args.encoder_name}{suffix}_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder","stage","dataset","horizon","mse","mae"])
        writer.writeheader(); writer.writerows(results)
    print(f"Results → {csv_path}")


if __name__ == "__main__":
    main()
