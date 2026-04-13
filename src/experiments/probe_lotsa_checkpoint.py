"""
Linear-Probe Evaluation of LOTSA-Pretrained Checkpoint
=======================================================
Loads the frozen encoders from a LOTSA CLIP pre-training run and evaluates
forecasting quality via a linear probe on the ICML benchmark datasets.

Protocol
--------
  1. Load time-series encoder + visual encoder from --checkpoint_dir (best weights).
  2. Freeze both encoders.
  3. For each dataset × horizon, train a linear head for --probe_epochs epochs.
  4. Report MSE / MAE; log every epoch to Comet ML.

Usage
-----
  python src/experiments/probe_lotsa_checkpoint.py \
      --checkpoint_dir checkpoints/ts_encoder_lotsa_20260329_2224 \
      --config src/configs/lotsa_clip.yaml \
      --data_dir ICML_datasets \
      --probe_epochs 20 \
      --results_dir results/probe_lotsa
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as torchF

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import (
    prepare_sequence,
    reshape_multivariate_series,
)
from time_series_loader import TimeSeriesDataModule

PROBE_DATASETS: List[str] = [
    "ETTm1.csv", "ETTm2.csv", "ETTh1.csv", "ETTh2.csv",
    "weather.csv", "traffic.csv", "electricity.csv", "exchange_rate.csv",
]
HORIZONS: List[int] = [96, 192, 336, 720]


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_encoders(checkpoint_dir: Path, config, device: torch.device):
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)

    enc_ckpt = checkpoint_dir / "time_series_best.pt"
    vis_ckpt = checkpoint_dir / "visual_encoder_best.pt"

    if enc_ckpt.exists():
        enc_state = torch.load(enc_ckpt, map_location=device)
        encoder.load_state_dict(
            enc_state.get("model_state_dict", enc_state.get("model_state", enc_state))
        )
        print(f"  Loaded encoder       : {enc_ckpt}")
    else:
        print(f"  [INFO] No temporal encoder checkpoint found — using random weights")

    if vis_ckpt.exists():
        vis_state = torch.load(vis_ckpt, map_location=device)
        raw = vis_state.get("model_state_dict", vis_state.get("model_state", vis_state))
        # Strip "encoder." prefix only if the checkpoint was saved with an outer wrapper
        # (i.e. all keys share the "encoder." prefix — no "output_proj" at top level).
        if all(k.startswith("encoder.") for k in raw):
            raw = {k[len("encoder."):]: v for k, v in raw.items()}
        visual.load_state_dict(raw)
        print(f"  Loaded visual encoder: {vis_ckpt}")
    else:
        print(f"  [INFO] No visual encoder checkpoint found — using random weights")

    encoder.eval()
    visual.eval()
    for p in list(encoder.parameters()) + list(visual.parameters()):
        p.requires_grad_(False)

    print(f"Loaded encoder from  : {enc_ckpt}")
    print(f"Loaded visual encoder: {vis_ckpt}")
    return encoder, visual


# ── embedding helpers ─────────────────────────────────────────────────────────

def _embed(encoder, visual, batch, device):
    """Embed a batch keeping each channel as an independent sample.

    For multivariate input (B, L, C), reshape_multivariate_series produces
    (B*C, 1, L) — each channel treated as a separate univariate series,
    matching exactly how the encoder was pre-trained on univariate LOTSA data.

    Returns:
        z:  (B*C, 2D) embeddings — one per (sample, channel)
        C:  number of channels (1 for univariate)
    """
    seq = batch[0].to(device).float()
    seq = prepare_sequence(seq)           # (B, L, C)
    C = seq.shape[2]
    x = reshape_multivariate_series(seq)  # (B*C, 1, L)
    with torch.no_grad():
        ze = encoder(x)   # (B*C, D)
        zv = visual(x)    # (B*C, D)
    return torch.cat([ze, zv], dim=1), C  # (B*C, 2D), C


def _collect(encoder, visual, loader, device):
    """Collect embeddings and targets, each channel as an independent sample.

    Returns:
        Z: (N*C, 2D) — one embedding per (sample, channel)
        Y: (N*C, L)  — corresponding univariate target sequence
    """
    zs, ys = [], []
    for batch in loader:
        z, C = _embed(encoder, visual, batch, device)
        zs.append(z)
        # y shape: (B, L, C) or (B, L) — expand channels to independent rows
        y = batch[1].to(device).float() if len(batch) > 1 else batch[0].to(device).float()
        if y.ndim == 3:
            B, L, Cy = y.shape
            y = y.permute(0, 2, 1).reshape(B * Cy, L)  # (B*C, L)
        ys.append(y)
    if not zs:
        return None, None
    return torch.cat(zs), torch.cat(ys)


# ── linear probe ──────────────────────────────────────────────────────────────

def probe_evaluate(
    encoder: nn.Module,
    visual: nn.Module,
    data_dir: Path,
    dataset_csv: str,
    horizons: List[int],
    device: torch.device,
    probe_epochs: int = 20,
    batch_size: int = 64,
    experiment=None,
    dataset_tag: str = "",
    scaler_type: str = "standard",
    few_shot_fraction: float = 1.0,
    rng: Optional[torch.Generator] = None,
) -> Dict[int, Dict[str, float]]:
    # Resolve the actual directory containing this CSV (handles subdirectories
    # like ETT-small/, weather/, traffic/, etc.)
    resolved_dir = str(data_dir)
    for candidate in Path(data_dir).rglob(dataset_csv):
        resolved_dir = str(candidate.parent)
        break

    max_horizon = max(horizons)
    try:
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
            sample_size=(96, 0, max_horizon),
            scaler_type=scaler_type,
        )
        module.setup()
    except Exception as exc:
        print(f"  [SKIP] {dataset_csv} — {exc}")
        return {}
    if not module.train_loaders:
        print(f"  [SKIP] {dataset_csv} not found under {data_dir}")
        return {}

    train_loader = module.train_loaders[0]
    test_loader = module.test_loaders[0] if module.test_loaders else None

    Z_tr, Y_tr = _collect(encoder, visual, train_loader, device)
    if Z_tr is None:
        print(f"  [SKIP] {dataset_csv} — train loader returned no batches")
        return {}

    # Few-shot subsampling: keep `few_shot_fraction` of training embeddings.
    # Encoder stays frozen — only the linear head sees fewer examples.
    if few_shot_fraction < 1.0:
        n_total = Z_tr.shape[0]
        n_keep = max(1, int(n_total * few_shot_fraction))
        idx = torch.randperm(n_total, generator=rng)[:n_keep]
        Z_tr = Z_tr[idx]
        Y_tr = Y_tr[idx]
        print(f"  Few-shot {few_shot_fraction*100:.0f}%: using {n_keep}/{n_total} train samples")

    feat_dim = Z_tr.shape[1]

    # Y_tr is now always (N*C, L) — each channel is an independent univariate sample
    # matching the encoder's training distribution (univariate LOTSA series)
    results: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        probe = nn.Linear(feat_dim, H).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        for ep in range(probe_epochs):
            probe.train()
            perm = torch.randperm(Z_tr.shape[0], device=device)
            ep_loss, n_batches = 0.0, 0
            for i in range(0, Z_tr.shape[0], batch_size):
                idx = perm[i: i + batch_size]
                z_b = Z_tr[idx]
                y_b = Y_tr[idx, :H]  # (batch, H) — univariate per channel
                if y_b.shape[1] < H:
                    continue
                loss = torchF.mse_loss(probe(z_b), y_b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += loss.item(); n_batches += 1

            if experiment is not None and n_batches > 0:
                step_key = f"{dataset_tag}/H{H}/probe_train_loss"
                experiment.log_metric(step_key, ep_loss / n_batches, step=ep + 1)

        if test_loader is not None:
            Z_test, Y_test = _collect(encoder, visual, test_loader, device)
            if Z_test is None:
                results[H] = {"mse": float("nan"), "mae": float("nan")}
                continue
            probe.eval()
            with torch.no_grad():
                y_eval = Y_test[:, :H]   # (N*C, H)
                if y_eval.shape[1] < H:
                    results[H] = {"mse": float("nan"), "mae": float("nan")}
                    continue
                pred = probe(Z_test)     # (N*C, H)
                mse = torchF.mse_loss(pred, y_eval).item()
                mae = (pred - y_eval).abs().mean().item()
            results[H] = {"mse": mse, "mae": mae}
            if experiment is not None:
                experiment.log_metric(f"{dataset_tag}/H{H}/mse", mse)
                experiment.log_metric(f"{dataset_tag}/H{H}/mae", mae)
        else:
            results[H] = {"mse": float("nan"), "mae": float("nan")}

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Linear-probe evaluation of a LOTSA-pretrained checkpoint"
    )
    p.add_argument("--checkpoint_dir", type=Path, required=True,
                   help="Directory containing time_series_best.pt and visual_encoder_best.pt")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--probe_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "probe_lotsa")
    p.add_argument("--datasets", nargs="+", default=PROBE_DATASETS,
                   help="CSV filenames to evaluate")
    p.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scaler_type", type=str, default="standard", choices=["standard", "minmax"])
    p.add_argument("--no_comet", action="store_true")
    p.add_argument(
        "--few_shot_fraction", type=float, default=1.0,
        help="Fraction of training data to use for the linear head (e.g. 0.01 for 1%%). "
             "Encoder stays frozen. Use 1.0 for standard full linear probe.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    few_shot_fraction = float(args.few_shot_fraction)
    if few_shot_fraction <= 0.0 or few_shot_fraction > 1.0:
        raise ValueError("--few_shot_fraction must be in (0, 1]")
    rng = torch.Generator()
    rng.manual_seed(args.seed)

    fraction_tag = "full" if few_shot_fraction == 1.0 else f"{few_shot_fraction*100:.0f}pct"
    print(f"Few-shot mode: {fraction_tag}  (fraction={few_shot_fraction})")

    # ── Comet ML ──────────────────────────────────────────────────────────────
    experiment: Optional[Any] = None
    if not args.no_comet:
        try:
            from comet_utils import create_comet_experiment
            experiment = create_comet_experiment("probe_lotsa_checkpoint")
            experiment.log_parameters({
                "checkpoint_dir": str(args.checkpoint_dir),
                "probe_epochs": args.probe_epochs,
                "batch_size": args.batch_size,
                "horizons": args.horizons,
                "datasets": args.datasets,
                "seed": args.seed,
                "few_shot_fraction": few_shot_fraction,
            })
            print("Comet ML experiment created.")
        except Exception as exc:
            print(f"[WARN] Comet ML init failed ({exc}). Running without logging.")

    # ── load frozen encoders ──────────────────────────────────────────────────
    encoder, visual = load_encoders(args.checkpoint_dir, config, device)

    # ── run probe per dataset ─────────────────────────────────────────────────
    rows: List[Dict] = []
    t_total = time.time()

    for ds in args.datasets:
        ds_tag = ds.replace(".csv", "").replace(".txt", "")
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_tag}")
        print(f"{'='*60}")

        metrics = probe_evaluate(
            encoder, visual,
            data_dir=args.data_dir,
            dataset_csv=ds,
            horizons=args.horizons,
            device=device,
            probe_epochs=args.probe_epochs,
            batch_size=args.batch_size,
            experiment=experiment,
            dataset_tag=ds_tag,
            scaler_type=args.scaler_type,
            few_shot_fraction=few_shot_fraction,
            rng=rng,
        )

        for H, m in metrics.items():
            rows.append({
                "dataset": ds_tag,
                "horizon": H,
                "mse": f"{m['mse']:.4f}",
                "mae": f"{m['mae']:.4f}",
            })
            print(f"  H={H:3d}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}")

    print(f"\nTotal elapsed: {(time.time() - t_total) / 60:.1f} min")

    # ── save CSV ──────────────────────────────────────────────────────────────
    out_csv = args.results_dir / f"probe_results_{fraction_tag}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "horizon", "mse", "mae"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out_csv}")

    if experiment is not None:
        experiment.end()
        print("Comet ML experiment ended.")


if __name__ == "__main__":
    main()
