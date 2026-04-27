"""
MoP Partial-Unfreeze Experiment.

Phase 1 — Warmup (10 epochs):
  Encoder fully frozen. Only MoP + prediction heads are updated.

Phase 2 — Partial unfreeze (40 epochs):
  Last 2 Mamba blocks + final_norm + output_proj of BOTH encoders are unfrozen.
  Three parameter groups with increasing LR:
    - MoP + heads           : lr_base
    - encoder last layers   : lr_base * 0.1   (fine-grained, don't corrupt pre-training)
  LR schedule: linear warm-up for 5 epochs then cosine decay to eta_min.

Goal: check whether giving the encoder a small amount of in-domain signal
from the ICML benchmarks improves zero-shot / few-shot transferability.
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
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent
root_dir   = src_dir.parent
if str(src_dir)  not in sys.path: sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path: sys.path.insert(0, str(root_dir))

import training_utils as tu
from models.mop_forecast import MoPForecastModel
from time_series_loader import TimeSeriesDataModule

ICML_DATASETS = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv",
]
HORIZONS    = [96, 192, 336, 720]
WARMUP_EP   = 10
UNFREEZE_EP = 40
TOTAL_EP    = WARMUP_EP + UNFREEZE_EP


def parse_args():
    p = argparse.ArgumentParser("MoP Partial-Unfreeze")
    p.add_argument("--checkpoint_dir",  type=Path, required=True)
    p.add_argument("--config",          type=Path, default=src_dir / "configs" / "lotsa_simclr_bimodal_nano.yaml")
    p.add_argument("--data_dir",        type=Path, default=root_dir / "ICML_datasets")
    p.add_argument("--results_dir",     type=Path, default=root_dir / "results" / "mop_tuning")
    p.add_argument("--batch_size",      type=int,  default=64)
    p.add_argument("--lr_base",         type=float, default=1e-3)
    p.add_argument("--lr_encoder_mult", type=float, default=0.1,
                   help="Multiplier applied to lr_base for unfrozen encoder layers")
    p.add_argument("--num_last_blocks", type=int,  default=2,
                   help="How many last Mamba blocks to unfreeze per encoder")
    p.add_argument("--num_prompts",     type=int,  default=8)
    p.add_argument("--hidden_dim",      type=int,  default=512)
    p.add_argument("--context_length",  type=int,  default=336)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    return p.parse_args()


def resolve_dir(data_dir, ds_name):
    for c in data_dir.rglob(ds_name):
        return str(c.parent)
    return str(data_dir)


def build_loaders(data_dir, context_length, max_horizon, batch_size, train=True, test=False):
    loaders = {}
    for ds_name in ICML_DATASETS:
        tag      = ds_name.replace(".csv", "")
        resolved = resolve_dir(data_dir, ds_name)
        try:
            module = TimeSeriesDataModule(
                dataset_name=ds_name, data_dir=resolved,
                batch_size=batch_size, val_batch_size=batch_size,
                num_workers=0, pin_memory=False, normalize=True,
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


def load_encoders(checkpoint_dir, config, device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual  = tu.build_visual_encoder_from_config(config.model).to(device)
    enc_path = checkpoint_dir / "time_series_best.pt"
    if not enc_path.exists(): enc_path = checkpoint_dir / "time_series_encoder.pt"
    vis_path = checkpoint_dir / "visual_encoder_best.pt"
    if not vis_path.exists(): vis_path = checkpoint_dir / "visual_encoder.pt"

    def _load(p):
        d = torch.load(p, map_location=device)
        return d.get("model_state_dict", d.get("model_state", d))

    encoder.load_state_dict(_load(enc_path))
    visual.load_state_dict(_load(vis_path))
    return encoder, visual


def freeze_encoder(enc):
    for p in enc.parameters():
        p.requires_grad = False


def unfreeze_last_layers(enc, num_last_blocks):
    """Unfreeze the last `num_last_blocks` Mamba blocks + final_norm + output_proj."""
    unfrozen_params = []
    total_blocks = len(enc.blocks)
    for i, block in enumerate(enc.blocks):
        if i >= total_blocks - num_last_blocks:
            for p in block.parameters():
                p.requires_grad = True
                unfrozen_params.append(p)
    for layer in [enc.final_norm, enc.output_proj]:
        for p in layer.parameters():
            p.requires_grad = True
            unfrozen_params.append(p)
    return unfrozen_params


def get_unfrozen_encoder_params(enc, num_last_blocks):
    """
    Works for both MambaEncoder and UpperTriDiagRPEncoder architectures.
    MambaEncoder      : enc.blocks, enc.final_norm, enc.output_proj
    UpperTriDiagRPEncoder : enc.encoder.blocks, enc.encoder.norm, enc.output_proj
    """
    params = []

    # Resolve where blocks and norm live
    if hasattr(enc, "blocks"):
        blocks   = enc.blocks
        norm     = getattr(enc, "final_norm", None)
    elif hasattr(enc, "encoder") and hasattr(enc.encoder, "blocks"):
        blocks   = enc.encoder.blocks
        norm     = getattr(enc.encoder, "norm", None)
    else:
        raise AttributeError(f"Cannot find 'blocks' in encoder type {type(enc).__name__}")

    total = len(blocks)
    for i, block in enumerate(blocks):
        if i >= total - num_last_blocks:
            params += list(block.parameters())

    if norm is not None:
        params += list(norm.parameters())

    params += list(enc.output_proj.parameters())
    return params


def linear_warmup_cosine(optimizer, warmup_steps, total_steps, eta_min_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def run_epoch(mop_model, loader_pool, iters, train_loaders, optimizer,
              scheduler, args, device, epoch_label):
    mop_model.train()
    total_loss, n_valid = 0.0, 0
    t0 = time.time()
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

        y_tgt  = y_in[:, :, :h_target]
        pred   = mop_model(x_in, h_target)
        y_tgt_ch = y_tgt.permute(0, 2, 1).reshape(B * C, h_target, 1)
        loss   = F.mse_loss(pred, y_tgt_ch)

        if torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in mop_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_valid    += 1

    scheduler.step()
    avg = total_loss / max(1, n_valid)
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    print(f"  {epoch_label} | Loss: {avg:.4f} | LRs: {[f'{l:.2e}' for l in lrs]} | {time.time()-t0:.1f}s")
    return avg


def evaluate(mop_model, test_loaders, args, device):
    results = []
    mop_model.eval()
    with torch.no_grad():
        for ds_name in ICML_DATASETS:
            tag = ds_name.replace(".csv", "")
            if tag not in test_loaders:
                continue
            print(f"  {tag}:")
            for H in HORIZONS:
                all_preds, all_trues = [], []
                for batch in test_loaders[tag]:
                    x = batch[0].to(device).float()
                    y = batch[1].to(device).float()
                    if y.shape[1] < H:
                        continue
                    B, L, C = x.shape
                    x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
                    pred = mop_model.greedy_predict(x_in, H, args.context_length)
                    y_tgt = y[:, :H, :].permute(0, 2, 1).reshape(B * C, H, 1)
                    all_preds.append(pred)
                    all_trues.append(y_tgt)
                if not all_preds:
                    continue
                pt  = torch.cat(all_preds)
                tt  = torch.cat(all_trues)
                mse = torch.mean((pt - tt) ** 2).item()
                mae = torch.mean(torch.abs(pt - tt)).item()
                print(f"    H={H}: MSE={mse:.4f} MAE={mae:.4f}")
                results.append({"dataset": tag, "horizon": H, "mse": mse, "mae": mae})
    return results


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)

    print("Loading encoders...")
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)

    # Fully freeze both encoders at start
    freeze_encoder(encoder)
    freeze_encoder(visual)

    enc_dim = getattr(encoder, "embedding_dim", config.model.get("embedding_dim", 128))
    mop_model = MoPForecastModel(
        encoder=encoder, visual_encoder=visual,
        input_dim=enc_dim * 2, hidden_dim=args.hidden_dim,
        num_prompts=args.num_prompts, horizons=HORIZONS,
        target_features=1, freeze_encoders=False,  # we manage freezing manually
    ).to(device)

    max_horizon = max(HORIZONS)
    print("Building ICML train loaders...")
    train_loaders = build_loaders(args.data_dir, args.context_length, max_horizon, args.batch_size, train=True)
    print(f"  {len(train_loaders)} loaders: {list(train_loaders.keys())}")
    loader_pool = list(train_loaders.items())
    iters = {tag: iter(loader) for tag, loader in train_loaders.items()}

    # ------------------------------------------------------------------ #
    # PHASE 1: Warmup — MoP + heads only                                  #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Phase 1: Warmup ({WARMUP_EP} epochs) — encoder fully frozen")
    print(f"{'='*60}")

    mop_params = list(mop_model.mop.parameters()) + list(mop_model.heads.parameters())
    opt1 = optim.AdamW(mop_params, lr=args.lr_base, weight_decay=1e-4)
    sch1 = CosineAnnealingLR(opt1, T_max=WARMUP_EP, eta_min=args.lr_base * 0.05)

    for ep in range(1, WARMUP_EP + 1):
        run_epoch(mop_model, loader_pool, iters, train_loaders, opt1, sch1, args, device,
                  f"Warmup Ep {ep:2d}/{WARMUP_EP}")

    print("\n--- Zero-shot after warmup ---")
    test_loaders = build_loaders(args.data_dir, args.context_length, max_horizon, args.batch_size, test=True)
    warmup_results = evaluate(mop_model, test_loaders, args, device)

    # ------------------------------------------------------------------ #
    # PHASE 2: Partial unfreeze                                            #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Phase 2: Partial unfreeze ({UNFREEZE_EP} epochs)")
    print(f"  Unfreezing last {args.num_last_blocks} blocks + final_norm + output_proj")
    print(f"  LR encoder = {args.lr_base * args.lr_encoder_mult:.2e}  |  LR MoP = {args.lr_base:.2e}")
    print(f"{'='*60}")

    enc_params = get_unfrozen_encoder_params(encoder, args.num_last_blocks)
    vis_params = get_unfrozen_encoder_params(visual,  args.num_last_blocks)
    for p in enc_params + vis_params:
        p.requires_grad = True

    n_enc = sum(p.numel() for p in enc_params + vis_params)
    n_mop = sum(p.numel() for p in mop_params)
    print(f"  Trainable — MoP+heads: {n_mop:,}  |  encoder last layers: {n_enc:,}")

    param_groups = [
        {"params": mop_params,             "lr": args.lr_base},
        {"params": enc_params + vis_params, "lr": args.lr_base * args.lr_encoder_mult},
    ]
    opt2 = optim.AdamW(param_groups, weight_decay=1e-4)
    sch2 = linear_warmup_cosine(opt2, warmup_steps=5, total_steps=UNFREEZE_EP, eta_min_ratio=0.05)

    # reset iterators
    iters = {tag: iter(loader) for tag, loader in train_loaders.items()}

    for ep in range(1, UNFREEZE_EP + 1):
        run_epoch(mop_model, loader_pool, iters, train_loaders, opt2, sch2, args, device,
                  f"Unfreeze Ep {ep:2d}/{UNFREEZE_EP}")

    print("\n--- Zero-shot after partial unfreeze ---")
    final_results = evaluate(mop_model, test_loaders, args, device)

    # Save checkpoint
    torch.save({"mop_model": mop_model.state_dict()},
               args.results_dir / "mop_partial_unfreeze_latest.pt")

    # Save CSV — both phases
    def save_csv(rows, path):
        with open(path, "w") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
            w.writeheader(); w.writerows(rows)

    save_csv(warmup_results, args.results_dir / "mop_partial_unfreeze_warmup_results.csv")
    save_csv(final_results,  args.results_dir / "mop_partial_unfreeze_final_results.csv")
    print(f"\nDone. Results saved to {args.results_dir}")


if __name__ == "__main__":
    main()
