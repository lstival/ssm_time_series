"""
Diagnose NaN divergence in CLIP Full training.

Investigates:
  1. Gradient norms per module (explosion?)
  2. Activation norms (encoder output)
  3. Logit scale / temperature (softmax saturation?)
  4. Loss per batch (where does NaN first appear?)
  5. AMP scaler overflow (skipped steps?)
  6. Input data statistics (NaN/Inf in batches?)
  7. Weight norm evolution

Run for ~2 epochs at full-size config, logging everything to a CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import util as u
import training_utils as tu
from cosine_training import _build_time_series_loaders


# ─────────────────────────────────────────────────────────────────────────────
def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)


def weight_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += p.detach().norm(2).item() ** 2
    return math.sqrt(total)


def activation_stats(t: torch.Tensor):
    """Return (mean, std, max_abs, has_nan, has_inf) for a tensor."""
    d = t.detach().float()
    return (
        d.mean().item(),
        d.std().item(),
        d.abs().max().item(),
        bool(torch.isnan(d).any()),
        bool(torch.isinf(d).any()),
    )


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results_dir", default="results/diagnose_clip_nan", type=Path)
    parser.add_argument("--epochs", default=12, type=int,
                        help="Run enough epochs to cross the NaN boundary (~epoch 10)")
    parser.add_argument("--max_batches", default=200, type=int,
                        help="Max batches per epoch (for speed); 0 = unlimited")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--grad_clip", default=0.0, type=float,
                        help="If >0, apply gradient clipping (test whether it fixes NaN)")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = tu.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AMP: {args.use_amp}  |  grad_clip: {args.grad_clip}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = _build_time_series_loaders(
        args.config, cfg.data, seed=args.seed
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model_cfg = cfg.model
    training_cfg = cfg.training
    temperature = float(training_cfg.get("temperature", 0.2))
    lr = float(training_cfg.get("learning_rate", 3e-4))
    wd = float(training_cfg.get("weight_decay", 1e-4))

    encoder = tu.build_encoder_from_config(model_cfg)
    visual_encoder = tu.build_visual_encoder_from_config(model_cfg)
    projection_dim = int(model_cfg.get("embedding_dim", 128))
    proj_head = u.build_projection_head(encoder, output_dim=projection_dim)
    vis_proj_head = u.build_projection_head(visual_encoder, output_dim=projection_dim)
    params = (list(encoder.parameters()) + list(visual_encoder.parameters()) +
              list(proj_head.parameters()) + list(vis_proj_head.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    warmup_epochs = int(training_cfg.get("warmup_epochs", 4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6
    )

    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")

    # ── CSV log ─────────────────────────────────────────────────────────────
    log_path = args.results_dir / "batch_diagnostics.csv"
    epoch_path = args.results_dir / "epoch_summary.csv"

    batch_fields = [
        "epoch", "batch",
        "input_mean", "input_std", "input_max", "input_nan", "input_inf",
        "q_mean", "q_std", "q_max", "q_nan", "q_inf",
        "k_mean", "k_std", "k_max", "k_nan", "k_inf",
        "logit_mean", "logit_std", "logit_max", "logit_nan", "logit_inf",
        "loss", "loss_nan",
        "grad_norm_enc", "grad_norm_vis", "grad_norm_proj",
        "scaler_scale", "lr",
    ]
    epoch_fields = [
        "epoch", "train_loss", "val_loss",
        "weight_norm_enc", "weight_norm_vis",
        "weight_norm_proj", "weight_norm_vis_proj",
        "lr",
    ]

    with open(log_path, "w", newline="") as bf, open(epoch_path, "w", newline="") as ef:
        bw = csv.DictWriter(bf, fieldnames=batch_fields)
        ew = csv.DictWriter(ef, fieldnames=epoch_fields)
        bw.writeheader()
        ew.writeheader()

        for epoch in range(args.epochs):
            encoder.train(); visual_encoder.train()
            proj_head.train(); vis_proj_head.train()

            epoch_loss = 0.0
            batches = 0
            first_nan_batch = None

            for b_idx, batch in enumerate(train_loader):
                if args.max_batches > 0 and b_idx >= args.max_batches:
                    break

                row = {"epoch": epoch + 1, "batch": b_idx + 1}

                # ── Extract inputs ───────────────────────────────────────
                if isinstance(batch, dict) and "target" in batch:
                    padded = batch["target"].to(device).float()
                    lengths = batch["lengths"].to(device)
                    L = int(lengths[0].item()) if (lengths == lengths[0]).all() else int(lengths.min().item())
                    x_q = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
                else:
                    seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
                    x_q = u.reshape_multivariate_series(seq)

                x_k = u.make_positive_view(x_q + float(training_cfg.get("noise_std", 0.001)) * torch.randn_like(x_q))

                # Input stats
                im, is_, ix, inan, iinf = activation_stats(x_q)
                row.update(dict(input_mean=im, input_std=is_, input_max=ix,
                                input_nan=inan, input_inf=iinf))

                # ── Forward ─────────────────────────────────────────────
                with autocast(enabled=args.use_amp and device.type == "cuda"):
                    q_enc = encoder(x_q)
                    k_enc = visual_encoder(x_k)
                    q_proj = F.normalize(proj_head(q_enc), dim=1)
                    k_proj = F.normalize(vis_proj_head(k_enc), dim=1)

                    logits = torch.matmul(q_proj, k_proj.T) / temperature
                    targets = torch.arange(logits.size(0), device=device)
                    loss = 0.5 * (F.cross_entropy(logits, targets) +
                                  F.cross_entropy(logits.T, targets))

                # Activation stats
                qm, qs, qx, qnan, qinf = activation_stats(q_proj)
                km, ks, kx, knan, kinf = activation_stats(k_proj)
                lm, ls, lx, lnan, linf = activation_stats(logits)
                row.update(dict(q_mean=qm, q_std=qs, q_max=qx, q_nan=qnan, q_inf=qinf,
                                k_mean=km, k_std=ks, k_max=kx, k_nan=knan, k_inf=kinf,
                                logit_mean=lm, logit_std=ls, logit_max=lx,
                                logit_nan=lnan, logit_inf=linf))

                loss_val = float(loss.item())
                loss_nan = bool(not math.isfinite(loss_val))
                row.update(dict(loss=loss_val if math.isfinite(loss_val) else -1,
                                loss_nan=loss_nan))

                if loss_nan and first_nan_batch is None:
                    first_nan_batch = b_idx + 1
                    print(f"  ⚠️  First NaN loss at epoch {epoch+1} batch {b_idx+1}")
                    print(f"     logit_max={lx:.4f}  q_nan={qnan}  k_nan={knan}")
                    print(f"     input_nan={inan}  input_inf={iinf}  input_max={ix:.4f}")

                # ── Backward ────────────────────────────────────────────
                optimizer.zero_grad(set_to_none=True)
                if args.use_amp:
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(params, args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    row["scaler_scale"] = scaler.get_scale()
                else:
                    if math.isfinite(loss_val):
                        loss.backward()
                    else:
                        # Still try backward to see if grads are NaN before loss is
                        try:
                            loss.backward()
                        except Exception:
                            pass
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(params, args.grad_clip)
                    optimizer.step()
                    row["scaler_scale"] = 1.0

                row["grad_norm_enc"] = grad_norm(encoder)
                row["grad_norm_vis"] = grad_norm(visual_encoder)
                row["grad_norm_proj"] = grad_norm(proj_head)
                row["lr"] = optimizer.param_groups[0]["lr"]

                if math.isfinite(loss_val):
                    epoch_loss += loss_val
                    batches += 1

                bw.writerow(row)

            # ── End of epoch ────────────────────────────────────────────
            scheduler.step()
            train_loss = epoch_loss / batches if batches > 0 else float("nan")

            val_loss = float("nan")
            if val_loader is not None:
                encoder.eval(); visual_encoder.eval()
                proj_head.eval(); vis_proj_head.eval()
                vl, vb = 0.0, 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        if isinstance(vbatch, dict) and "target" in vbatch:
                            padded = vbatch["target"].to(device).float()
                            lengths = vbatch["lengths"].to(device)
                            L = int(lengths[0].item()) if (lengths == lengths[0]).all() else int(lengths.min().item())
                            vx_q = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
                        else:
                            seq = u.prepare_sequence(u.extract_sequence(vbatch)).to(device).float()
                            vx_q = u.reshape_multivariate_series(seq)
                        vx_k = u.make_positive_view(vx_q)
                        vq = F.normalize(proj_head(encoder(vx_q)), dim=1)
                        vk = F.normalize(vis_proj_head(visual_encoder(vx_k)), dim=1)
                        vlogits = torch.matmul(vq, vk.T) / temperature
                        vtargets = torch.arange(vlogits.size(0), device=device)
                        vloss = 0.5 * (F.cross_entropy(vlogits, vtargets) +
                                       F.cross_entropy(vlogits.T, vtargets))
                        v = float(vloss.item())
                        if math.isfinite(v):
                            vl += v; vb += 1
                val_loss = vl / vb if vb > 0 else float("nan")

            print(f"Epoch {epoch+1}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
                  + (f"  first_nan_batch={first_nan_batch}" if first_nan_batch else ""))

            ew.writerow({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "weight_norm_enc": weight_norm(encoder),
                "weight_norm_vis": weight_norm(visual_encoder),
                "weight_norm_proj": weight_norm(proj_head),
                "weight_norm_vis_proj": weight_norm(vis_proj_head),
                "lr": optimizer.param_groups[0]["lr"],
            })
            bf.flush(); ef.flush()

    print(f"\n✅ Diagnostics saved to {args.results_dir}")
    print(f"   Batch log : {log_path}")
    print(f"   Epoch log : {epoch_path}")

    # ── Quick summary ────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(log_path)
    nan_rows = df[df["loss_nan"] == True]
    print(f"\n=== NaN batches: {len(nan_rows)} / {len(df)} total ===")
    if len(nan_rows):
        print(nan_rows[["epoch","batch","input_nan","q_nan","k_nan",
                         "logit_max","grad_norm_enc","grad_norm_vis"]].head(20).to_string())

    # Gradient norm evolution (per epoch mean)
    print("\n=== Grad norm (encoder) per epoch ===")
    print(df.groupby("epoch")["grad_norm_enc"].agg(["mean","max"]).to_string())

    print("\n=== Logit max per epoch ===")
    print(df.groupby("epoch")["logit_max"].agg(["mean","max"]).to_string())


if __name__ == "__main__":
    main()
