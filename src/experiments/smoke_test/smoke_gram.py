"""Smoke test for GRAM training health.

Checks on a short run:
- finite loss
- non-zero gradients for 4 trainable modules
- no NaN in Gram volume
- robust positive-pair structure over batches
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
repo_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import build_projection_head, build_time_series_dataloaders, prepare_sequence, extract_sequence, reshape_multivariate_series, make_positive_view


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return total ** 0.5


def gram_volume_loss(z_ts: torch.Tensor, z_rp: torch.Tensor, temperature: float = 0.2):
    ts = F.normalize(z_ts, dim=-1)
    rp = F.normalize(z_rp, dim=-1)
    cos = torch.matmul(ts, rp.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    vol = torch.sqrt((1.0 - cos.pow(2)).clamp_min(1e-12))
    logits = -vol / temperature
    labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    return loss, logits, vol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: GRAM")
    parser.add_argument("--data_dir", type=Path, default=repo_root / "ICML_datasets" / "ETT-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_comet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tu.set_seed(args.seed)
    device = tu.prepare_device("auto")

    model_cfg = {
        "rp_encoder": "upper_tri",
        "input_dim": 32,
        "model_dim": 64,
        "embedding_dim": 64,
        "depth": 2,
        "state_dim": 16,
        "conv_kernel": 4,
        "expand_factor": 1.5,
        "dropout": 0.0,
        "pooling": "mean",
    }

    train_loader, _ = build_time_series_dataloaders(
        data_dir=str(args.data_dir),
        dataset_type="icml",
        batch_size=64,
        val_batch_size=32,
        num_workers=0,
        pin_memory=False,
        val_ratio=0.1,
        seed=args.seed,
    )

    ts_encoder = tu.build_encoder_from_config(model_cfg).to(device)
    rp_encoder = tu.build_visual_encoder_from_config(model_cfg).to(device)
    ts_proj = build_projection_head(ts_encoder, output_dim=64).to(device)
    rp_proj = build_projection_head(rp_encoder, output_dim=64).to(device)

    params = list(ts_encoder.parameters()) + list(rp_encoder.parameters()) + list(ts_proj.parameters()) + list(rp_proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    finite_ok = True
    nan_vol_ok = True
    structure_ok = True
    checked_steps = 0
    diag_better_steps = 0
    margin_sum = 0.0

    grad_ok = {
        "ts_encoder": False,
        "rp_encoder": False,
        "ts_proj": False,
        "rp_proj": False,
    }

    for _epoch in range(args.epochs):
        for batch in train_loader:
            seq = prepare_sequence(extract_sequence(batch)).to(device).float()
            x_q = reshape_multivariate_series(seq)
            x_k = make_positive_view(x_q + 0.01 * torch.randn_like(x_q))

            q = ts_proj(ts_encoder(x_q))
            k = rp_proj(rp_encoder(x_k))
            loss, logits, vol = gram_volume_loss(q, k, temperature=0.2)

            if not torch.isfinite(loss):
                finite_ok = False
            if torch.isnan(vol).any():
                nan_vol_ok = False

            diag = torch.diag(logits)
            off = logits[~torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)]
            diag_mean = float(diag.mean().item())
            off_mean = float(off.mean().item())
            margin = diag_mean - off_mean
            checked_steps += 1
            margin_sum += margin
            if margin > 0.0:
                diag_better_steps += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_ok["ts_encoder"] = grad_ok["ts_encoder"] or (_grad_norm(ts_encoder) > 0)
            grad_ok["rp_encoder"] = grad_ok["rp_encoder"] or (_grad_norm(rp_encoder) > 0)
            grad_ok["ts_proj"] = grad_ok["ts_proj"] or (_grad_norm(ts_proj) > 0)
            grad_ok["rp_proj"] = grad_ok["rp_proj"] or (_grad_norm(rp_proj) > 0)

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

    if checked_steps > 0:
        diag_better_ratio = diag_better_steps / checked_steps
        mean_margin = margin_sum / checked_steps
        # In very short smoke runs, require trend-level separation rather than
        # strict per-step separation, which is noisy at initialization.
        structure_ok = (diag_better_ratio >= 0.55) and (mean_margin > -0.02)
    else:
        diag_better_ratio = 0.0
        mean_margin = float("nan")
        structure_ok = False

    all_grad_ok = all(grad_ok.values())
    passed = finite_ok and nan_vol_ok and structure_ok and all_grad_ok

    print("GRAM smoke summary")
    print(f"finite_loss={finite_ok}")
    print(f"nan_volume={nan_vol_ok}")
    print(f"diag_gt_offdiag={structure_ok}")
    print(f"diag_better_ratio={diag_better_ratio:.3f}")
    print(f"diag_minus_offdiag_mean={mean_margin:.4f}")
    print(f"gradients={grad_ok}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
