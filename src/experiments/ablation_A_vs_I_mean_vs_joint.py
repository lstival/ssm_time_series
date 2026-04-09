"""
Ablation A vs. Ablation I: Direct Comparison Test
==================================================
Compares "mean" strategy (Ablation A winner) vs "joint"/"global_l2" strategy
(Ablation I winner for high-d) on the SAME model architecture and training setup.

Research Question:
  Does global_l2 (joint RP strategy) outperform mean aggregation
  on multivariate datasets, especially at high dimensionality?

Protocol:
  1. Load the Ablation A "mean" checkpoint as a shared starting point.
  2. Finetune two copies for --finetune_epochs: one keeping "mean" strategy,
     one switching to "joint" (global_l2) strategy.
  3. Freeze encoders; linear-probe on ETTm1, Weather, Traffic for horizons
     [96, 192, 336, 720].
  4. Compare MSE/MAE across datasets by dimensionality:
     - ETTm1 (7 vars):      `mean` should win (low-d)
     - Weather (21 vars):   either could win (medium-d)
     - Traffic (321 vars):  `joint` should win (high-d)

Usage:
  python src/experiments/ablation_A_vs_I_mean_vs_joint.py \
      --config src/configs/lotsa_clip.yaml \
      --checkpoint_dir results/ablation_A/strategy_mean \
      --finetune_epochs 10 \
      --probe_epochs 30 \
      --results_dir results/mean_vs_joint
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF

# ── path setup ──────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import (
    build_time_series_dataloaders,
    clip_contrastive_loss,
    make_positive_view,
    prepare_sequence,
    reshape_multivariate_series,
    build_projection_head,
)
from models.mamba_encoder import MambaEncoder
from models.mamba_visual_encoder import MambaVisualEncoder
from time_series_loader import TimeSeriesDataModule

# ── constants ────────────────────────────────────────────────────────────────
STRATEGIES: List[str] = ["mean", "joint"]
PROBE_DATASETS: List[str] = ["ETTm1.csv", "weather.csv", "traffic.csv"]
HORIZONS: List[int] = [96, 192, 336, 720]

# (csv_name, label, n_channels, expected_winner)
DATASETS_INFO = [
    ("ETTm1.csv",   "Low-D (7 vars)",    7,   "mean"),
    ("weather.csv", "Medium-D (21 vars)", 21,  "either"),
    ("traffic.csv", "High-D (321 vars)",  321, "joint"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_encoders(
    model_cfg: Dict,
    rp_mv_strategy: str,
) -> Tuple[MambaEncoder, MambaVisualEncoder]:
    encoder = tu.build_encoder_from_config(model_cfg)
    visual = tu.build_visual_encoder_from_config(
        model_cfg,
        rp_mode="correct",
        rp_mv_strategy=rp_mv_strategy,
        repr_type="rp",
    )
    return encoder, visual


def _load_checkpoint(
    encoder: nn.Module,
    visual: nn.Module,
    checkpoint_dir: Path,
) -> None:
    """Load encoder and visual encoder weights from an Ablation A checkpoint dir."""
    enc_path = checkpoint_dir / "encoder.pt"
    vis_path = checkpoint_dir / "visual_encoder.pt"

    if enc_path.exists():
        state = torch.load(enc_path, map_location="cpu")
        encoder.load_state_dict(state["model_state"], strict=False)
        print(f"  Loaded encoder from {enc_path}")
    else:
        print(f"  [WARN] No encoder checkpoint at {enc_path}, starting from scratch.")

    if vis_path.exists():
        state = torch.load(vis_path, map_location="cpu")
        visual.load_state_dict(state["model_state"], strict=False)
        print(f"  Loaded visual encoder from {vis_path}")
    else:
        print(f"  [WARN] No visual encoder checkpoint at {vis_path}, starting from scratch.")


def _clip_train_epoch(
    encoder: nn.Module,
    visual: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.01,
) -> float:
    encoder.train(); visual.train(); proj.train(); vproj.train()
    total, n = 0.0, 0
    for batch in loader:
        if isinstance(batch, dict) and "target" in batch:
            seq = batch["target"].to(device).float()
            if "lengths" in batch:
                seq = seq[:, : int(batch["lengths"].max().item())]
        elif isinstance(batch, (tuple, list)):
            seq = batch[0].to(device).float()
        else:
            seq = batch.to(device).float()

        seq = prepare_sequence(seq)
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))

        q = torchF.normalize(proj(encoder(x_q)), dim=1)
        k = torchF.normalize(vproj(visual(x_k)), dim=1)
        loss = clip_contrastive_loss(q, k)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()); n += 1

    return total / max(1, n)


@torch.no_grad()
def _clip_val_epoch(
    encoder: nn.Module,
    visual: nn.Module,
    proj: nn.Module,
    vproj: nn.Module,
    loader,
    device: torch.device,
    noise_std: float = 0.01,
) -> float:
    encoder.eval(); visual.eval(); proj.eval(); vproj.eval()
    total, n = 0.0, 0
    for batch in loader:
        if isinstance(batch, dict) and "target" in batch:
            seq = batch["target"].to(device).float()
            if "lengths" in batch:
                seq = seq[:, : int(batch["lengths"].max().item())]
        elif isinstance(batch, (tuple, list)):
            seq = batch[0].to(device).float()
        else:
            seq = batch.to(device).float()
        seq = prepare_sequence(seq)
        x_q = reshape_multivariate_series(seq)
        x_k = make_positive_view(x_q + noise_std * torch.randn_like(x_q))
        q = torchF.normalize(proj(encoder(x_q)), dim=1)
        k = torchF.normalize(vproj(visual(x_k)), dim=1)
        total += float(clip_contrastive_loss(q, k).item()); n += 1
    encoder.train(); visual.train(); proj.train(); vproj.train()
    return total / max(1, n)


class _LinearProbe(nn.Module):
    def __init__(self, feat_dim: int, horizon: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def _probe_evaluate(
    encoder: nn.Module,
    visual: nn.Module,
    data_dir: Path,
    dataset_csv: str,
    horizons: List[int],
    device: torch.device,
    probe_epochs: int = 30,
    batch_size: int = 64,
) -> Dict[int, Dict[str, float]]:
    """Train linear probes and return {horizon: {mse, mae}}."""
    max_horizon = max(horizons)
    module = TimeSeriesDataModule(
        dataset_name=dataset_csv,
        data_dir=str(data_dir),
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        normalize=True,
        train=True,
        val=False,
        test=True,
        sample_size=(96, 0, max_horizon),
    )
    module.setup()
    if not module.train_loaders:
        return {}
    train_loader = module.train_loaders[0]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    encoder.eval(); visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    def _embed(batch):
        seq = batch[0].to(device).float()
        seq = prepare_sequence(seq)
        x = reshape_multivariate_series(seq)
        with torch.no_grad():
            ze = encoder(x)
            zv = visual(x)
        return torch.cat([ze, zv], dim=1)

    def _collect(loader):
        zs, ys = [], []
        for batch in loader:
            z = _embed(batch)
            y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
            zs.append(z); ys.append(y)
        return torch.cat(zs), torch.cat(ys)

    Z_tr, Y_tr = _collect(train_loader)
    feat_dim = Z_tr.shape[1]

    results: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        probe = _LinearProbe(feat_dim, H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(probe_epochs):
            probe.train()
            perm = torch.randperm(Z_tr.shape[0], device=device)
            for i in range(0, Z_tr.shape[0], 64):
                idx = perm[i: i + 64]
                z_b = Z_tr[idx]; y_b = Y_tr[idx]
                if y_b.ndim == 3:
                    y_b = y_b[:, :H, 0]
                elif y_b.ndim == 2:
                    y_b = y_b[:, :H]
                if y_b.shape[1] < H:
                    continue
                pred = probe(z_b)
                loss = torchF.mse_loss(pred, y_b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        if test_loader is not None:
            Z_test, Y_test = _collect(test_loader)
            probe.eval()
            with torch.no_grad():
                pred = probe(Z_test)
                if Y_test.ndim == 3: Y_test = Y_test[:, :H, 0]
                elif Y_test.ndim == 2: Y_test = Y_test[:, :H]
                if Y_test.shape[1] < H:
                    results[H] = {"mse": float("nan"), "mae": float("nan")}
                    continue
                mse = torchF.mse_loss(pred, Y_test).item()
                mae = (pred - Y_test).abs().mean().item()
            results[H] = {"mse": mse, "mae": mae}
        else:
            results[H] = {"mse": float("nan"), "mae": float("nan")}

    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(True)

    return results


def save_results(
    results: Dict[str, Dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for strategy in STRATEGIES:
        if strategy not in results:
            continue
        for ds, _, _, _ in DATASETS_INFO:
            if ds not in results[strategy]:
                continue
            row = {"Strategy": strategy, "Dataset": ds}
            for H in HORIZONS:
                if H in results[strategy][ds]:
                    m = results[strategy][ds][H]
                    row[f"H{H}_MSE"] = m.get("mse", float("nan"))
                    row[f"H{H}_MAE"] = m.get("mae", float("nan"))
            rows.append(row)

    if rows:
        fieldnames = ["Strategy", "Dataset"] + [
            f"H{h}_{metric}" for h in HORIZONS for metric in ["MSE", "MAE"]
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {output_path}")


def analyze_results(results: Dict[str, Dict]) -> None:
    print(f"\n{'='*70}")
    print("COMPARISON: MEAN vs. JOINT")
    print(f"{'='*70}\n")

    for ds, ds_label, n_ch, expected in DATASETS_INFO:
        print(f"Dataset: {ds_label}: {ds}  (expected winner: {expected})")
        print(f"  {'Horizon':<10} {'Mean MSE':<15} {'Joint MSE':<15} {'Winner':<10}")
        print(f"  {'-'*52}")

        winners = {"mean": 0, "joint": 0}
        mean_mse_list, joint_mse_list = [], []

        for H in HORIZONS:
            m_mse = results.get("mean", {}).get(ds, {}).get(H, {}).get("mse")
            j_mse = results.get("joint", {}).get(ds, {}).get(H, {}).get("mse")

            if m_mse is not None and j_mse is not None and not (np.isnan(m_mse) or np.isnan(j_mse)):
                winner = "mean" if m_mse < j_mse else "joint"
                winners[winner] += 1
                mean_mse_list.append(m_mse)
                joint_mse_list.append(j_mse)
                print(f"  {H:<10} {m_mse:<15.4f} {j_mse:<15.4f} {winner:<10}")
            else:
                print(f"  {H:<10} {'N/A':<15} {'N/A':<15}")

        if mean_mse_list and joint_mse_list:
            mean_avg = np.mean(mean_mse_list)
            joint_avg = np.mean(joint_mse_list)
            diff = (mean_avg - joint_avg) / mean_avg * 100
            print(f"\n  Avg MSE → mean={mean_avg:.4f}  joint={joint_avg:.4f}  "
                  f"diff={diff:+.1f}% ({'joint wins' if diff > 0 else 'mean wins'})")
            print(f"  Horizons won: mean={winners['mean']}  joint={winners['joint']}")
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation: mean vs joint RP strategy, finetuned from Ablation A checkpoint"
    )
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--checkpoint_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_A" / "strategy_mean",
                   help="Ablation A checkpoint to start from (encoder.pt + visual_encoder.pt)")
    p.add_argument("--finetune_epochs", type=int, default=10,
                   help="CLIP finetuning epochs per strategy (starting from checkpoint)")
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "mean_vs_joint")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets",
                   help="Data dir for linear probe evaluation")
    p.add_argument("--pretrain_data_dir", type=Path,
                   default=src_dir.parent / "data",
                   help="Data dir for CLIP finetuning")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("Ablation: Mean vs Joint RP Strategy (finetuned from Ablation A checkpoint)")
    print(f"{'='*70}")
    print(f"Device:          {device}")
    print(f"Checkpoint:      {args.checkpoint_dir}")
    print(f"Finetune epochs: {args.finetune_epochs}")
    print(f"Probe epochs:    {args.probe_epochs}")
    print(f"Results dir:     {args.results_dir}")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cfg = tu.load_config(args.config)

    train_loader, val_loader = build_time_series_dataloaders(
        data_dir=str(args.pretrain_data_dir),
        dataset_name=cfg.data.get("dataset_name", ""),
        dataset_type=cfg.data.get("dataset_type", "icml"),
        batch_size=int(cfg.data.get("batch_size", 128)),
        val_batch_size=int(cfg.data.get("val_batch_size", 64)),
        num_workers=int(cfg.data.get("num_workers", 4)),
        pin_memory=bool(cfg.data.get("pin_memory", True)),
        val_ratio=float(cfg.data.get("val_ratio", 0.1)),
        cronos_kwargs=dict(cfg.data.get("cronos_kwargs", {})),
        seed=args.seed,
    )

    all_results: Dict[str, Dict] = {}

    for strategy in STRATEGIES:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*70}")

        tu.set_seed(args.seed)
        encoder, visual = _build_encoders(cfg.model, rp_mv_strategy=strategy)
        encoder.to(device); visual.to(device)

        # Load shared starting checkpoint
        _load_checkpoint(encoder, visual, args.checkpoint_dir)

        proj = build_projection_head(encoder).to(device)
        vproj = build_projection_head(visual).to(device)

        params = (list(encoder.parameters()) + list(visual.parameters())
                  + list(proj.parameters()) + list(vproj.parameters()))
        optimizer = torch.optim.AdamW(
            params,
            lr=float(cfg.training.get("learning_rate", 1e-3)),
            weight_decay=float(cfg.training.get("weight_decay", 1e-4)),
        )
        noise_std = float(cfg.training.get("noise_std", 0.01))

        t0 = time.time()
        for ep in range(args.finetune_epochs):
            train_loss = _clip_train_epoch(
                encoder, visual, proj, vproj, train_loader, optimizer, device,
                noise_std=noise_std,
            )
            val_loss: Optional[float] = None
            if val_loader is not None:
                val_loss = _clip_val_epoch(
                    encoder, visual, proj, vproj, val_loader, device, noise_std=noise_std,
                )
            val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
            print(f"  epoch {ep+1:02d}/{args.finetune_epochs}  train={train_loss:.4f}{val_str}")

        elapsed = time.time() - t0
        print(f"  Finetuning done — {elapsed:.0f}s")

        # Save finetuned checkpoint
        ckpt_out = args.results_dir / f"strategy_{strategy}"
        ckpt_out.mkdir(exist_ok=True)
        torch.save({"model_state": encoder.state_dict()}, ckpt_out / "encoder.pt")
        torch.save({"model_state": visual.state_dict()}, ckpt_out / "visual_encoder.pt")

        # Linear probe
        strategy_results: Dict[str, Dict] = {}
        for ds, ds_label, _, _ in DATASETS_INFO:
            print(f"\n  Probing {ds} ({ds_label})...")
            metrics = _probe_evaluate(
                encoder, visual,
                data_dir=args.data_dir,
                dataset_csv=ds,
                horizons=HORIZONS,
                device=device,
                probe_epochs=args.probe_epochs,
            )
            strategy_results[ds] = metrics
            for H, m in metrics.items():
                print(f"    H={H:3d}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

        all_results[strategy] = strategy_results

    save_results(all_results, args.results_dir / "mean_vs_joint_comparison.csv")
    analyze_results(all_results)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
