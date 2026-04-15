"""
SSL Method Forecast Comparison — Visual Evaluation
===================================================
For each dataset × horizon, produces a multi-panel figure showing the same
test window for all 4 nano SSL methods side-by-side:

  - Lookback context  : solid green
  - Forecast (pred)   : dashed orange
  - Ground truth      : dotted dark-green

One figure per dataset, one row per method, one column per horizon.
Figures saved to results/ssl_forecast_plots/.

Usage
-----
  python src/experiments/plot_ssl_comparison.py [--horizons 96 192 336 720]
                                                 [--datasets ETTm1 weather ...]
                                                 [--n_examples 3]
                                                 [--probe_epochs 20]
                                                 [--context_length 96]
                                                 [--out_dir results/ssl_forecast_plots]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── make src importable ───────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

import training_utils as tu
from util import prepare_sequence, reshape_multivariate_series
from time_series_loader import TimeSeriesDataModule

# ── constants ─────────────────────────────────────────────────────────────────

METHODS: Dict[str, Dict] = {
    "CLIP": {
        "checkpoint": "/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/clip_nano/ts_clip_nano_lotsa_20260409_181021",
        "config":     "src/configs/lotsa_clip_nano.yaml",
    },
    "GRAM": {
        "checkpoint": "/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/gram_nano/ts_gram_nano_lotsa_20260409_181020",
        "config":     "src/configs/lotsa_gram_nano.yaml",
    },
    "VL-JEPA": {
        "checkpoint": "/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/vl_jepa_nano/ts_vl_jepa_nano_lotsa_20260409_181014",
        "config":     "src/configs/lotsa_vl_jepa_nano.yaml",
    },
    "SimCLR": {
        "checkpoint": "/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/simclr_bimodal_nano/ts_simclr_bimodal_nano_lotsa_20260409_182831",
        "config":     "src/configs/lotsa_simclr_bimodal_nano.yaml",
    },
}

ALL_DATASETS = [
    "ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv",
    "electricity.csv", "exchange_rate.csv", "traffic.csv", "weather.csv",
]

DATA_DIR    = Path("/home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets")
PROJECT_ROOT = Path("/home/WUR/stiva001/WUR/ssm_time_series")

# ── colour palette ─────────────────────────────────────────────────────────────
C_LOOKBACK = "#2ca02c"       # solid green
C_PRED     = "#ff7f0e"       # dashed orange
C_GT       = "#1a6b1a"       # dotted dark-green

METHOD_COLORS = {
    "CLIP":    "#1f77b4",
    "GRAM":    "#d62728",
    "VL-JEPA": "#9467bd",
    "SimCLR":  "#e377c2",
}


# ── checkpoint loading ─────────────────────────────────────────────────────────

def load_encoders(checkpoint_dir: Path, config, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual  = tu.build_visual_encoder_from_config(config.model).to(device)

    enc_ckpt = checkpoint_dir / "time_series_best.pt"
    vis_ckpt = checkpoint_dir / "visual_encoder_best.pt"

    enc_state = torch.load(enc_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(
        enc_state.get("model_state_dict", enc_state.get("model_state", enc_state))
    )

    if vis_ckpt.exists():
        vis_state = torch.load(vis_ckpt, map_location=device, weights_only=False)
        raw = vis_state.get("model_state_dict", vis_state.get("model_state", vis_state))
        # Only strip encoder. prefix if ALL keys have it (outer-wrapper format)
        if all(k.startswith("encoder.") for k in raw):
            raw = {k[len("encoder."):]: v for k, v in raw.items()}
        visual.load_state_dict(raw)

    encoder.eval(); visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)
    return encoder, visual


# ── embedding ──────────────────────────────────────────────────────────────────

def _embed(encoder, visual, seq_batch, device):
    """seq_batch: (B, T, C) or (B, C, T) tensor from loader."""
    x = seq_batch.to(device).float()
    x = prepare_sequence(x)
    x = reshape_multivariate_series(x)
    with torch.no_grad():
        ze = encoder(x)
        zv = visual(x)
    return torch.cat([ze, zv], dim=1)


def collect_embeddings(encoder, visual, loader, device):
    Zs, Ys, Xs = [], [], []
    for batch in loader:
        x_raw = batch[0].to(device).float()
        z = _embed(encoder, visual, x_raw, device)
        y = batch[1].to(device).float() if len(batch) > 1 else x_raw
        Zs.append(z); Ys.append(y); Xs.append(x_raw)
    return torch.cat(Zs), torch.cat(Ys), torch.cat(Xs)


# ── data loading ───────────────────────────────────────────────────────────────

def build_loaders(dataset_csv: str, context_length: int, horizon: int, batch_size: int = 64):
    resolved_dir = str(DATA_DIR)
    for candidate in DATA_DIR.rglob(dataset_csv):
        resolved_dir = str(candidate.parent)
        break

    module = TimeSeriesDataModule(
        dataset_name=dataset_csv,
        data_dir=resolved_dir,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        normalize=True,
        train=True,
        val=False,
        test=True,
        sample_size=(context_length, 0, horizon),
        scaler_type="standard",
    )
    module.setup()
    train_loader = module.train_loaders[0] if module.train_loaders else None
    test_loader  = module.test_loaders[0]  if module.test_loaders  else None

    # Extract inverse_transform from the underlying dataset (for denormalisation)
    inverse_fn = None
    if module.dataset_loaders:
        entry = module.dataset_loaders[0]
        for ds_attr in ("test_dataset", "train_dataset"):
            ds = getattr(entry, ds_attr, None)
            fn = getattr(ds, "inverse_transform", None)
            if fn is not None:
                inverse_fn = fn
                break

    return train_loader, test_loader, inverse_fn


# ── linear probe ───────────────────────────────────────────────────────────────

def train_probe(Z_tr, Y_tr, horizon: int, probe_epochs: int, batch_size: int, device) -> nn.Linear:
    feat_dim = Z_tr.shape[1]
    probe = nn.Linear(feat_dim, horizon).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3)
    for _ in range(probe_epochs):
        probe.train()
        perm = torch.randperm(Z_tr.shape[0], device=device)
        for i in range(0, Z_tr.shape[0], batch_size):
            idx = perm[i: i + batch_size]
            z_b = Z_tr[idx]
            y_b = Y_tr[idx]
            if y_b.ndim == 3:
                y_b = y_b[:, :horizon, 0]
            elif y_b.ndim == 2:
                y_b = y_b[:, :horizon]
            if y_b.shape[1] < horizon:
                continue
            loss = F.mse_loss(probe(z_b), y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    probe.eval()
    return probe


# ── main pipeline per method × dataset × horizon ──────────────────────────────

def _denorm(arr_2d: np.ndarray, inverse_fn, n_features: int, ch: int = 0) -> np.ndarray:
    """Inverse-transform channel `ch` from a (N, T) normalised array.

    inverse_transform expects (N, n_features) — we broadcast channel `ch`
    across all features, inverse-transform, then extract channel `ch` back.
    """
    if inverse_fn is None:
        return arr_2d
    N, T = arr_2d.shape
    out = np.zeros_like(arr_2d)
    for i in range(N):
        dummy = np.zeros((T, n_features), dtype=np.float32)
        dummy[:, ch] = arr_2d[i]
        restored = inverse_fn(dummy)   # (T, n_features)
        out[i] = restored[:, ch]
    return out


def run_method(
    method_name: str,
    checkpoint_dir: Path,
    config_path: Path,
    dataset_csv: str,
    horizons: List[int],
    n_examples: int,
    probe_epochs: int,
    context_length: int,
    device: torch.device,
) -> Dict[int, Dict]:
    """Returns {horizon: {lookback, pred, gt, mse, lookback_d, pred_d, gt_d}} for n_examples."""
    config = tu.load_config(config_path)
    encoder, visual = load_encoders(checkpoint_dir, config, device)

    results = {}
    for H in horizons:
        print(f"    H={H} ...", end=" ", flush=True)
        try:
            train_loader, test_loader, inverse_fn = build_loaders(dataset_csv, context_length, H)
        except Exception as e:
            print(f"LOAD ERROR: {e}")
            results[H] = None
            continue

        if train_loader is None or test_loader is None:
            print("no loader")
            results[H] = None
            continue

        # Collect train embeddings + labels
        Z_tr, Y_tr, _ = collect_embeddings(encoder, visual, train_loader, device)
        if Y_tr.ndim == 3:
            Y_tr_1d = Y_tr[:, :H, 0]
        else:
            Y_tr_1d = Y_tr[:, :H]

        probe = train_probe(Z_tr, Y_tr_1d, H, probe_epochs, 64, device)

        # Collect test samples for plotting
        Z_te, Y_te, X_te = collect_embeddings(encoder, visual, test_loader, device)
        if Y_te.ndim == 3:
            Y_te_1d = Y_te[:, :H, 0]
            n_features = Y_te.shape[2]
        else:
            Y_te_1d = Y_te[:, :H]
            n_features = 1
        if X_te.ndim == 3:
            X_te_1d = X_te[:, :, 0]   # (N, T)
            n_features = max(n_features, X_te.shape[2])
        else:
            X_te_1d = X_te

        with torch.no_grad():
            pred = probe(Z_te[:n_examples])   # (n_examples, H)

        mse = F.mse_loss(probe(Z_te), Y_te_1d).item()
        print(f"MSE={mse:.4f}")

        lb_np  = X_te_1d[:n_examples].cpu().numpy()
        pr_np  = pred.cpu().numpy()
        gt_np  = Y_te_1d[:n_examples].cpu().numpy()

        results[H] = {
            # normalised (standard-scaled)
            "lookback": lb_np,
            "pred":     pr_np,
            "gt":       gt_np,
            "mse":      mse,
            # denormalised (original scale)
            "lookback_d": _denorm(lb_np, inverse_fn, n_features),
            "pred_d":     _denorm(pr_np, inverse_fn, n_features),
            "gt_d":       _denorm(gt_np, inverse_fn, n_features),
        }
    return results


# ── plotting ───────────────────────────────────────────────────────────────────

def _draw_panels(axes, methods, horizons, all_results, ex, keys, ylabel):
    """Fill a grid of axes with lookback/gt/pred for one scale variant."""
    for row, method in enumerate(methods):
        for col, H in enumerate(horizons):
            ax = axes[row, col]
            data = all_results[method].get(H)

            if data is None:
                ax.axis("off")
                ax.set_title(f"{method} H={H}\nN/A", fontsize=9)
                continue

            lb  = data[keys["lb"]][ex]
            pr  = data[keys["pr"]][ex]
            gt  = data[keys["gt"]][ex]
            mse = data["mse"]

            T = len(lb)
            t_lb   = np.arange(T)
            t_fore = np.arange(T, T + H)

            ax.plot(t_lb,   lb, color=C_LOOKBACK, lw=1.4,  alpha=0.85)
            ax.plot(t_fore, gt, color=C_GT,       lw=1.6,  linestyle=":",  alpha=0.9)
            ax.plot(t_fore, pr, color=C_PRED,     lw=1.6,  linestyle="--", alpha=0.9)
            ax.axvline(x=T - 0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5)

            ax.set_title(f"{method}  H={H}\nMSE={mse:.4f}", fontsize=9,
                         fontweight="bold", color=METHOD_COLORS.get(method, "black"))
            ax.set_xlabel("Timestep", fontsize=8)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", linestyle=":", alpha=0.4)


def _save_fig(fig, out_path):
    handles = [
        mpatches.Patch(color=C_LOOKBACK, label="Lookback (context)"),
        plt.Line2D([0], [0], color=C_GT,   lw=1.5, linestyle=":",  label="Ground truth"),
        plt.Line2D([0], [0], color=C_PRED, lw=1.5, linestyle="--", label="Forecast"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dataset(
    dataset_csv: str,
    all_results: Dict[str, Dict[int, Dict]],
    horizons: List[int],
    n_examples: int,
    out_dir: Path,
):
    ds_label  = dataset_csv.replace(".csv", "")
    methods   = list(all_results.keys())
    n_methods = len(methods)
    n_horizons = len(horizons)
    figsize = (4.5 * n_horizons, 3.2 * n_methods)

    out_norm   = out_dir / "normalised"
    out_denorm = out_dir / "original_scale"
    out_norm.mkdir(parents=True, exist_ok=True)
    out_denorm.mkdir(parents=True, exist_ok=True)

    for ex in range(n_examples):
        # ── normalised figure ─────────────────────────────────────────────────
        fig, axes = plt.subplots(n_methods, n_horizons, figsize=figsize, squeeze=False)
        _draw_panels(axes, methods, horizons, all_results, ex,
                     keys={"lb": "lookback", "pr": "pred", "gt": "gt"},
                     ylabel="Norm. value (z-score)")
        fig.suptitle(
            f"{ds_label} — Normalised  (example {ex+1}/{n_examples})",
            fontsize=13, fontweight="bold", y=1.01,
        )
        _save_fig(fig, out_norm / f"{ds_label}_ex{ex+1}.png")

        # ── denormalised figure ───────────────────────────────────────────────
        fig, axes = plt.subplots(n_methods, n_horizons, figsize=figsize, squeeze=False)
        _draw_panels(axes, methods, horizons, all_results, ex,
                     keys={"lb": "lookback_d", "pr": "pred_d", "gt": "gt_d"},
                     ylabel="Original scale")
        fig.suptitle(
            f"{ds_label} — Original Scale  (example {ex+1}/{n_examples})",
            fontsize=13, fontweight="bold", y=1.01,
        )
        _save_fig(fig, out_denorm / f"{ds_label}_ex{ex+1}.png")


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SSL forecast comparison plots")
    p.add_argument("--horizons",        nargs="+", type=int,
                   default=[96, 192, 336, 720])
    p.add_argument("--datasets",        nargs="+",
                   default=[d.replace(".csv","") for d in ALL_DATASETS])
    p.add_argument("--n_examples",      type=int, default=3,
                   help="Test samples to plot per dataset")
    p.add_argument("--probe_epochs",    type=int, default=20)
    p.add_argument("--context_length",  type=int, default=96)
    p.add_argument("--out_dir",         default="results/ssl_forecast_plots")
    p.add_argument("--methods",         nargs="+", default=list(METHODS.keys()),
                   help="Subset of methods to include")
    return p.parse_args()


def main():
    args = parse_args()
    tu.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets_csv = [
        d if d.endswith(".csv") else d + ".csv"
        for d in args.datasets
    ]
    methods_to_run = {k: v for k, v in METHODS.items() if k in args.methods}

    for ds_csv in datasets_csv:
        ds_label = ds_csv.replace(".csv", "")
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_label}")
        print(f"{'='*70}")

        all_results: Dict[str, Dict] = {}
        for method_name, spec in methods_to_run.items():
            print(f"  [{method_name}]")
            checkpoint_dir = Path(spec["checkpoint"])
            config_path    = PROJECT_ROOT / spec["config"]

            if not checkpoint_dir.exists():
                print(f"    SKIP — checkpoint not found: {checkpoint_dir}")
                continue
            if not config_path.exists():
                print(f"    SKIP — config not found: {config_path}")
                continue

            try:
                all_results[method_name] = run_method(
                    method_name   = method_name,
                    checkpoint_dir= checkpoint_dir,
                    config_path   = config_path,
                    dataset_csv   = ds_csv,
                    horizons      = args.horizons,
                    n_examples    = args.n_examples,
                    probe_epochs  = args.probe_epochs,
                    context_length= args.context_length,
                    device        = device,
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

        if not all_results:
            print(f"  No results for {ds_label}, skipping plot.")
            continue

        plot_dataset(
            dataset_csv  = ds_csv,
            all_results  = all_results,
            horizons     = args.horizons,
            n_examples   = args.n_examples,
            out_dir      = out_dir,
        )

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
