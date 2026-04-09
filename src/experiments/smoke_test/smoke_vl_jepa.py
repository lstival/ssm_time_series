"""Smoke test for VL-JEPA training health.

Checks on a short run:
- finite loss
- non-zero gradients for 6 trainable modules
- target representation norm above collapse threshold
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent
repo_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from byol_training import MLPHead, _update_ema
from models.mamba_visual_encoder import UpperTriDiagRPEncoder
from util import build_time_series_dataloaders, prepare_sequence, extract_sequence, reshape_multivariate_series


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return total ** 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test: VL-JEPA")
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
    rp_encoder = UpperTriDiagRPEncoder(
        patch_len=int(model_cfg["input_dim"]),
        d_model=int(model_cfg["model_dim"]),
        n_layers=int(model_cfg["depth"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        rp_mv_strategy="mean",
    ).to(device)

    ts_projector = MLPHead(64, 128, 128).to(device)
    rp_projector = MLPHead(64, 128, 128).to(device)
    ts_predictor = MLPHead(128, 64, 128).to(device)
    rp_predictor = MLPHead(128, 64, 128).to(device)

    ts_encoder_ema = copy.deepcopy(ts_encoder).to(device)
    rp_encoder_ema = copy.deepcopy(rp_encoder).to(device)
    ts_projector_ema = copy.deepcopy(ts_projector).to(device)
    rp_projector_ema = copy.deepcopy(rp_projector).to(device)
    for p in list(ts_encoder_ema.parameters()) + list(rp_encoder_ema.parameters()) + list(ts_projector_ema.parameters()) + list(rp_projector_ema.parameters()):
        p.requires_grad_(False)

    params = (
        list(ts_encoder.parameters())
        + list(rp_encoder.parameters())
        + list(ts_projector.parameters())
        + list(rp_projector.parameters())
        + list(ts_predictor.parameters())
        + list(rp_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    finite_ok = True
    collapse_ok = True
    grad_ok = {
        "ts_encoder": False,
        "rp_encoder": False,
        "ts_projector": False,
        "rp_projector": False,
        "ts_predictor": False,
        "rp_predictor": False,
    }

    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            seq = prepare_sequence(extract_sequence(batch)).to(device).float()
            x = reshape_multivariate_series(seq)
            x = x + 0.05 * torch.randn_like(x)

            z_ts_online = ts_projector(ts_encoder(x))
            z_rp_online = rp_projector(rp_encoder(x))
            p_ts = ts_predictor(z_ts_online)
            p_rp = rp_predictor(z_rp_online)

            with torch.no_grad():
                z_rp_target = rp_projector_ema(rp_encoder_ema(x))
                z_ts_target = ts_projector_ema(ts_encoder_ema(x))

            loss = 0.5 * (
                torch.nn.functional.mse_loss(p_ts, z_rp_target)
                + torch.nn.functional.mse_loss(p_rp, z_ts_target)
            )

            if not torch.isfinite(loss):
                finite_ok = False

            z_norm = float(z_rp_target.norm(dim=-1).mean().item())
            if z_norm <= 0.01:
                collapse_ok = False

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_ok["ts_encoder"] = grad_ok["ts_encoder"] or (_grad_norm(ts_encoder) > 0)
            grad_ok["rp_encoder"] = grad_ok["rp_encoder"] or (_grad_norm(rp_encoder) > 0)
            grad_ok["ts_projector"] = grad_ok["ts_projector"] or (_grad_norm(ts_projector) > 0)
            grad_ok["rp_projector"] = grad_ok["rp_projector"] or (_grad_norm(rp_projector) > 0)
            grad_ok["ts_predictor"] = grad_ok["ts_predictor"] or (_grad_norm(ts_predictor) > 0)
            grad_ok["rp_predictor"] = grad_ok["rp_predictor"] or (_grad_norm(rp_predictor) > 0)

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            tau = 1.0 - (1.0 - 0.996) * ((1.0 + torch.cos(torch.tensor(torch.pi * global_step / total_steps)).item()) / 2.0)
            _update_ema(ts_encoder, ts_encoder_ema, tau)
            _update_ema(rp_encoder, rp_encoder_ema, tau)
            _update_ema(ts_projector, ts_projector_ema, tau)
            _update_ema(rp_projector, rp_projector_ema, tau)
            global_step += 1

    all_grad_ok = all(grad_ok.values())
    passed = finite_ok and collapse_ok and all_grad_ok

    print("VL-JEPA smoke summary")
    print(f"finite_loss={finite_ok}")
    print(f"collapse_check={collapse_ok}")
    print(f"gradients={grad_ok}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
