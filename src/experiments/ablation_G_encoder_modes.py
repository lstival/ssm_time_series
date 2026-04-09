"""
Ablation G — Per-Encoder-Mode Linear Probe
===========================================
Uses a single pretrained checkpoint and evaluates forecasting quality
when using only one encoder branch vs. both:

  Mode             | Input to linear probe
  -----------------|------------------------------------------
  temporal_only    | ze  = MambaEncoder(x)            (D)
  visual_only      | zv  = MambaVisualEncoder(x)       (D)
  multimodal       | [ze ‖ zv] (current default)       (2D)
  multimodal_mean  | (ze + zv) / 2                     (D)  ← same dim as single-encoder modes

Hypothesis (NeurIPS claim):
  - visual_only lower MSE at H=96  (RP captures local / periodic structure)
  - temporal_only lower MSE at H=720 (SSM captures long-range dynamics)
  - multimodal best at every horizon (branches are complementary)

Protocol
--------
  1. Load frozen encoders from --checkpoint_dir
     (or use random weights with --random_weights for smoke tests).
  2. Per dataset: collect embeddings ONCE for both encoders, then slice by mode.
  3. For each mode × horizon, train a separate linear head for --probe_epochs.
  4. Evaluate on SOTA test split; report MSE / MAE.

Usage
-----
  # Smoke test (random weights, fast)
  python src/experiments/ablation_G_encoder_modes.py \\
      --config src/experiments/smoke_test/smoke_config.yaml \\
      --random_weights \\
      --datasets ETTm1.csv \\
      --horizons 96 192 \\
      --probe_epochs 2 \\
      --results_dir results/smoke/ablation_G

  # Full run (pretrained checkpoint)
  python src/experiments/ablation_G_encoder_modes.py \\
      --checkpoint_dir checkpoints/ts_encoder_lotsa_latest \\
      --config src/configs/lotsa_clip.yaml \\
      --probe_epochs 20 \\
      --results_dir results/ablation_G
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torchF

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import prepare_sequence, reshape_multivariate_series
from time_series_loader import TimeSeriesDataModule

ENCODER_MODES: List[str] = ["temporal_only", "visual_only", "multimodal", "multimodal_mean"]
PROBE_DATASETS: List[str] = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv", "exchange_rate.csv",
]
HORIZONS: List[int] = [96, 192, 336, 720]


# ── embedding collection ──────────────────────────────────────────────────────

@torch.no_grad()
def _collect_both(
    encoder: nn.Module,
    visual: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Collect ze, zv, and labels from a loader in a single pass.

    Returns (Z_temporal, Z_visual, Y) tensors, or (None, None, None) if empty.
    """
    zts, zvs, ys = [], [], []
    for batch in loader:
        x = batch[0].to(device).float()
        x = reshape_multivariate_series(prepare_sequence(x))
        ze = encoder(x)
        zv = visual(x)
        zts.append(ze)
        zvs.append(zv)
        y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
        ys.append(y)
    if not zts:
        return None, None, None
    return torch.cat(zts), torch.cat(zvs), torch.cat(ys)


def _mode_embedding(Z_t: torch.Tensor, Z_v: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "temporal_only":
        return Z_t
    if mode == "visual_only":
        return Z_v
    if mode == "multimodal_mean":
        return (Z_t + Z_v) / 2.0  # (B, D) — same dim as single-encoder modes
    return torch.cat([Z_t, Z_v], dim=1)  # multimodal: (B, 2D)


# ── linear probe (per mode) ───────────────────────────────────────────────────

def _probe_one_horizon(
    Z_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    Z_test: Optional[torch.Tensor],
    Y_test: Optional[torch.Tensor],
    H: int,
    device: torch.device,
    probe_epochs: int,
    batch_size: int,
) -> Dict[str, float]:
    feat_dim = Z_tr.shape[1]
    probe = nn.Linear(feat_dim, H).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(probe_epochs):
        probe.train()
        perm = torch.randperm(Z_tr.shape[0], device=device)
        for i in range(0, Z_tr.shape[0], batch_size):
            idx = perm[i: i + batch_size]
            z_b = Z_tr[idx]
            y_b = Y_tr[idx]
            if y_b.ndim == 3:
                y_b = y_b[:, :H, 0]
            elif y_b.ndim == 2:
                y_b = y_b[:, :H]
            if y_b.shape[1] < H:
                continue
            loss = torchF.mse_loss(probe(z_b), y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    if Z_test is None or Y_test is None:
        return {"mse": float("nan"), "mae": float("nan")}

    probe.eval()
    with torch.no_grad():
        pred = probe(Z_test)
        if Y_test.ndim == 3:
            Y_test = Y_test[:, :H, 0]
        elif Y_test.ndim == 2:
            Y_test = Y_test[:, :H]
        if Y_test.shape[1] < H:
            return {"mse": float("nan"), "mae": float("nan")}
        mse = torchF.mse_loss(pred, Y_test).item()
        mae = (pred - Y_test).abs().mean().item()
    return {"mse": mse, "mae": mae}


def probe_dataset(
    encoder: nn.Module,
    visual: nn.Module,
    data_dir: Path,
    dataset_csv: str,
    modes: List[str],
    horizons: List[int],
    device: torch.device,
    probe_epochs: int,
    batch_size: int,
    experiment=None,
    ds_tag: str = "",
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Returns {mode: {horizon: {mse, mae}}} for one dataset."""
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
        print(f"    [SKIP] {dataset_csv} not found under {data_dir}")
        return {m: {} for m in modes}

    train_loader = module.train_loaders[0]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    encoder.eval(); visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    # Single forward pass for both encoders
    Zt_tr, Zv_tr, Y_tr = _collect_both(encoder, visual, train_loader, device)
    if Zt_tr is None:
        print(f"    [SKIP] {dataset_csv} — train loader returned no batches")
        return {m: {} for m in modes}

    Zt_te, Zv_te, Y_te = (None, None, None)
    if test_loader is not None:
        Zt_te, Zv_te, Y_te = _collect_both(encoder, visual, test_loader, device)

    results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for mode in modes:
        Z_tr = _mode_embedding(Zt_tr, Zv_tr, mode)
        Z_te = _mode_embedding(Zt_te, Zv_te, mode) if Zt_te is not None else None
        results[mode] = {}
        for H in horizons:
            m = _probe_one_horizon(
                Z_tr, Y_tr, Z_te, Y_te, H, device, probe_epochs, batch_size
            )
            results[mode][H] = m
            if experiment is not None:
                experiment.log_metric(f"{mode}/{ds_tag}/H{H}/mse", m["mse"])
                experiment.log_metric(f"{mode}/{ds_tag}/H{H}/mae", m["mae"])

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-encoder-mode linear probe — Ablation G"
    )
    p.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="Directory with time_series_best.pt + visual_encoder_best.pt")
    p.add_argument("--random_weights", action="store_true",
                   help="Use randomly initialised encoders (smoke test; no checkpoint needed)")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--probe_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_G")
    p.add_argument("--datasets", nargs="+", default=PROBE_DATASETS)
    p.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    p.add_argument("--encoder_modes", nargs="+", default=ENCODER_MODES,
                   choices=ENCODER_MODES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_comet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.random_weights and args.checkpoint_dir is None:
        raise ValueError("Provide --checkpoint_dir or use --random_weights for smoke testing")

    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ── Comet ML ──────────────────────────────────────────────────────────────
    experiment = None
    if not args.no_comet and not args.random_weights:
        try:
            from comet_utils import create_comet_experiment
            experiment = create_comet_experiment("ablation_G_encoder_modes")
            experiment.log_parameters({
                "checkpoint_dir": str(args.checkpoint_dir),
                "probe_epochs": args.probe_epochs,
                "horizons": args.horizons,
                "datasets": args.datasets,
                "encoder_modes": args.encoder_modes,
                "seed": args.seed,
            })
            print("Comet ML experiment created.")
        except Exception as exc:
            print(f"[WARN] Comet ML init failed ({exc}). Running without logging.")

    # ── build / load encoders ─────────────────────────────────────────────────
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)

    if args.random_weights:
        print("Using RANDOM weights (smoke-test mode — no checkpoint loaded).")
    else:
        enc_names = ("time_series_best.pt", "encoder.pt", "time_series_last.pt")
        vis_names = ("visual_encoder_best.pt", "visual_encoder.pt", "visual_encoder_last.pt")
        enc_ckpt = next(
            (args.checkpoint_dir / n for n in enc_names if (args.checkpoint_dir / n).exists()),
            args.checkpoint_dir / "time_series_best.pt",
        )
        vis_ckpt = next(
            (args.checkpoint_dir / n for n in vis_names if (args.checkpoint_dir / n).exists()),
            args.checkpoint_dir / "visual_encoder_best.pt",
        )
        if not enc_ckpt.exists():
            raise FileNotFoundError(f"Temporal encoder checkpoint not found: {enc_ckpt}")
        if not vis_ckpt.exists():
            raise FileNotFoundError(f"Visual encoder checkpoint not found: {vis_ckpt}")

        def _load(module, path):
            state = torch.load(path, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            module.load_state_dict(state)
            print(f"  Loaded {path.name}")

        _load(encoder, enc_ckpt)
        _load(visual, vis_ckpt)
        print(f"Checkpoint: {args.checkpoint_dir}")

    # ── run per dataset (embeddings collected once, probed per mode) ──────────
    rows = []
    t_total = time.time()

    for ds in args.datasets:
        ds_tag = ds.replace(".csv", "").replace(".txt", "")
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_tag}")
        print(f"{'='*60}")

        ds_results = probe_dataset(
            encoder, visual,
            data_dir=args.data_dir,
            dataset_csv=ds,
            modes=args.encoder_modes,
            horizons=args.horizons,
            device=device,
            probe_epochs=args.probe_epochs,
            batch_size=args.batch_size,
            experiment=experiment,
            ds_tag=ds_tag,
        )

        for mode, horizon_metrics in ds_results.items():
            for H, m in horizon_metrics.items():
                rows.append({
                    "mode": mode,
                    "dataset": ds_tag,
                    "horizon": H,
                    "mse": f"{m['mse']:.4f}",
                    "mae": f"{m['mae']:.4f}",
                })
                print(f"  [{mode:15s}]  H={H:3d}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    print(f"\nTotal elapsed: {(time.time() - t_total) / 60:.1f} min")

    # ── save CSV ──────────────────────────────────────────────────────────────
    out_csv = args.results_dir / "ablation_G_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out_csv}")

    if experiment is not None:
        experiment.end()
        print("Comet ML experiment ended.")


if __name__ == "__main__":
    main()
