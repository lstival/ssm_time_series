"""
Plot Prediction vs Ground Truth for Linear Probe Evaluation
===========================================================
Loads frozen encoders from a LOTSA checkpoint, trains a linear probe (H=96)
on each dataset, then plots predictions vs ground truth for the first 200
timesteps of channel 0 on the test set.

Two figures are produced:
  - Figure 1 (best):  ETTm1, weather
  - Figure 2 (worst): exchange_rate, electricity

Usage
-----
  python src/experiments/plot_probe_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF

# ── paths ─────────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import prepare_sequence, reshape_multivariate_series
from time_series_loader import TimeSeriesDataModule

CHECKPOINT_DIR = Path(
    "/home/WUR/stiva001/WUR/ssm_time_series/checkpoints/lotsa_ablation_best"
    "/ts_encoder_lotsa_ablation_best_20260403_1213"
)
CONFIG_PATH = Path(
    "/home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_ablation_best.yaml"
)
DATA_DIR = Path("/home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets")
RESULTS_DIR = Path(
    "/home/WUR/stiva001/WUR/ssm_time_series/results/probe_lotsa_ablation_best"
)

BEST_DATASETS  = ["ETTm1.csv", "weather.csv"]
WORST_DATASETS = ["exchange_rate.csv", "electricity.csv"]

HORIZON       = 96
PROBE_EPOCHS  = 20
BATCH_SIZE    = 64
TIMESTEPS_PLOT = 200   # first N timesteps of channel 0 to display


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_encoders(checkpoint_dir: Path, config, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual  = tu.build_visual_encoder_from_config(config.model).to(device)

    enc_ckpt = checkpoint_dir / "time_series_best.pt"
    vis_ckpt = checkpoint_dir / "visual_encoder_best.pt"

    enc_state = torch.load(enc_ckpt, map_location=device)
    vis_state = torch.load(vis_ckpt, map_location=device)

    encoder.load_state_dict(
        enc_state.get("model_state_dict", enc_state.get("model_state", enc_state))
    )
    visual.load_state_dict(
        vis_state.get("model_state_dict", vis_state.get("model_state", vis_state))
    )

    encoder.eval()
    visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    print(f"Loaded encoder       : {enc_ckpt}")
    print(f"Loaded visual encoder: {vis_ckpt}")
    return encoder, visual


# ── embedding helpers ─────────────────────────────────────────────────────────

def _embed(encoder, visual, batch, device):
    seq = batch[0].to(device).float()
    seq = prepare_sequence(seq)
    x = reshape_multivariate_series(seq)
    with torch.no_grad():
        ze = encoder(x)
        zv = visual(x)
    return torch.cat([ze, zv], dim=1)


def _collect(encoder, visual, loader, device):
    zs, ys = [], []
    for batch in loader:
        zs.append(_embed(encoder, visual, batch, device))
        y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
        ys.append(y)
    if not zs:
        return None, None
    return torch.cat(zs), torch.cat(ys)


# ── data loading ──────────────────────────────────────────────────────────────

def build_loaders(dataset_csv: str):
    """Return (train_loader, test_loader) for the given dataset CSV."""
    resolved_dir = str(DATA_DIR)
    for candidate in DATA_DIR.rglob(dataset_csv):
        resolved_dir = str(candidate.parent)
        break

    module = TimeSeriesDataModule(
        dataset_name=dataset_csv,
        data_dir=resolved_dir,
        batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        normalize=True,
        train=True,
        val=False,
        test=True,
        sample_size=(96, 0, HORIZON),
        scaler_type="standard",
    )
    module.setup()
    train_loader = module.train_loaders[0]
    test_loader  = module.test_loaders[0] if module.test_loaders else None
    return train_loader, test_loader


# ── linear probe training ─────────────────────────────────────────────────────

def train_probe(encoder, visual, train_loader, device):
    Z_tr, Y_tr = _collect(encoder, visual, train_loader, device)
    feat_dim = Z_tr.shape[1]
    probe = nn.Linear(feat_dim, HORIZON).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for ep in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(Z_tr.shape[0], device=device)
        ep_loss, n_b = 0.0, 0
        for i in range(0, Z_tr.shape[0], BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            z_b = Z_tr[idx]
            y_b = Y_tr[idx]
            if y_b.ndim == 3:
                y_b = y_b[:, :HORIZON, 0]
            elif y_b.ndim == 2:
                y_b = y_b[:, :HORIZON]
            if y_b.shape[1] < HORIZON:
                continue
            loss = torchF.mse_loss(probe(z_b), y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); n_b += 1
        if n_b > 0:
            print(f"  epoch {ep+1:3d}/{PROBE_EPOCHS}  loss={ep_loss/n_b:.5f}")

    probe.eval()
    return probe


# ── prediction collection ─────────────────────────────────────────────────────

def collect_predictions(encoder, visual, probe, test_loader, device):
    """
    Returns (pred_flat, gt_flat) as 1-D numpy arrays suitable for time-series
    plotting along the prediction horizon axis for channel 0.

    Strategy: concatenate horizon-step predictions from every test sample so
    that a single time-axis of length N * HORIZON is formed.  Channel 0 of the
    target is used when targets are multivariate.
    """
    Z_test, Y_test = _collect(encoder, visual, test_loader, device)
    with torch.no_grad():
        pred = probe(Z_test)            # (N, H)

    if Y_test.ndim == 3:
        Y_test = Y_test[:, :HORIZON, 0]
    elif Y_test.ndim == 2:
        Y_test = Y_test[:, :HORIZON]

    pred_np = pred.cpu().numpy().flatten()      # (N*H,)
    gt_np   = Y_test.cpu().numpy().flatten()    # (N*H,)

    mse = float(torchF.mse_loss(pred, Y_test).item())
    return pred_np, gt_np, mse


# ── plotting ──────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "ETTm1.csv":        "ETTm1",
    "weather.csv":      "Weather",
    "exchange_rate.csv":"Exchange Rate",
    "electricity.csv":  "Electricity",
}

PALETTE = {
    "gt":   "#2c7bb6",   # steel blue
    "pred": "#d7191c",   # brick red
}


def plot_figure(datasets, encoder, visual, device, fig_name: str):
    n_datasets = len(datasets)
    fig, axes = plt.subplots(
        n_datasets, 1,
        figsize=(12, 4 * n_datasets),
        squeeze=False,
    )

    for row, ds in enumerate(datasets):
        ax = axes[row, 0]
        ds_label = LABEL_MAP.get(ds, ds.replace(".csv", ""))
        print(f"\n{'='*60}")
        print(f"Processing: {ds_label}")

        try:
            train_loader, test_loader = build_loaders(ds)
        except Exception as exc:
            ax.set_title(f"{ds_label} — LOAD ERROR: {exc}")
            ax.axis("off")
            continue

        if test_loader is None:
            ax.set_title(f"{ds_label} — no test loader")
            ax.axis("off")
            continue

        probe = train_probe(encoder, visual, train_loader, device)
        pred_np, gt_np, mse = collect_predictions(
            encoder, visual, probe, test_loader, device
        )
        print(f"  Test MSE: {mse:.4f}")

        T = min(TIMESTEPS_PLOT, len(gt_np))
        t = np.arange(T)

        ax.plot(t, gt_np[:T],   color=PALETTE["gt"],   lw=1.5,
                label="Ground Truth", alpha=0.9)
        ax.plot(t, pred_np[:T], color=PALETTE["pred"],  lw=1.5,
                label="Prediction",  alpha=0.85, linestyle="--")

        ax.set_title(
            f"{ds_label}  |  H={HORIZON}  |  MSE = {mse:.4f}",
            fontsize=13, fontweight="bold", pad=8,
        )
        ax.set_xlabel("Timestep (flattened horizon)", fontsize=11)
        ax.set_ylabel("Normalised value", fontsize=11)
        ax.legend(fontsize=10, loc="upper right", framealpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Linear Probe — Prediction vs Ground Truth  (first {TIMESTEPS_PLOT} steps, channel 0)",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    out_path = RESULTS_DIR / fig_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    tu.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config  = tu.load_config(CONFIG_PATH)
    encoder, visual = load_encoders(CHECKPOINT_DIR, config, device)

    print("\n" + "="*60)
    print("Figure 1 — BEST datasets: ETTm1, Weather")
    print("="*60)
    plot_figure(
        BEST_DATASETS, encoder, visual, device,
        fig_name="fig1_best_datasets.png",
    )

    print("\n" + "="*60)
    print("Figure 2 — WORST datasets: Exchange Rate, Electricity")
    print("="*60)
    plot_figure(
        WORST_DATASETS, encoder, visual, device,
        fig_name="fig2_worst_datasets.png",
    )

    print("\nDone. Both figures saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
