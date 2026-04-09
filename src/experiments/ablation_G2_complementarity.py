"""
Ablation G2 — Complementarity Analysis
=======================================
Answers: does the visual branch contribute information the temporal encoder
does NOT already have?

Complementarity metric (per dataset × horizon):
  C = min(MSE_temporal, MSE_visual) − MSE_multimodal

  C > 0  → multimodal beats both single-encoder probes → branches are complementary
  C ≈ 0  → multimodal matches the stronger single encoder → no gain from fusion
  C < 0  → should not occur with a well-trained model (sanity-check flag)

Protocol
--------
  1. Load frozen encoders from --checkpoint_dir
     (or use random weights with --random_weights for smoke tests).
  2. Per dataset: collect ze, zv with a single forward pass (_collect_both).
  3. Train 3 linear probes: temporal_only (D), visual_only (D), multimodal (2D).
  4. Compute C per (dataset, horizon).
  5. Write CSV with columns:
       dataset, horizon, mse_temporal, mse_visual, mse_multimodal, complementarity

Usage
-----
  # Smoke test (random weights, fast)
  python src/experiments/ablation_G2_complementarity.py \\
      --random_weights \\
      --config src/experiments/smoke_test/smoke_config.yaml \\
      --datasets ETTm1.csv \\
      --horizons 96 192 \\
      --probe_epochs 2 \\
      --results_dir results/smoke/ablation_G2 \\
      --no_comet

  # Full run (pretrained checkpoint)
  python src/experiments/ablation_G2_complementarity.py \\
      --checkpoint_dir checkpoints/ts_encoder_lotsa_latest \\
      --config src/configs/lotsa_clip.yaml \\
      --probe_epochs 20 \\
      --results_dir results/ablation_G2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu

# Reuse all probe utilities from ablation_G — no code duplication
from ablation_G_encoder_modes import (
    _collect_both,
    probe_dataset,
    PROBE_DATASETS,
    HORIZONS,
)

ANALYSIS_MODES: List[str] = ["temporal_only", "visual_only", "multimodal"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complementarity analysis — Ablation G2"
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
                   default=src_dir.parent / "results" / "ablation_G2")
    p.add_argument("--datasets", nargs="+", default=PROBE_DATASETS)
    p.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
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
            experiment = create_comet_experiment("ablation_G2_complementarity")
            experiment.log_parameters({
                "checkpoint_dir": str(args.checkpoint_dir),
                "probe_epochs": args.probe_epochs,
                "horizons": args.horizons,
                "datasets": args.datasets,
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

    # ── run per dataset ───────────────────────────────────────────────────────
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
            modes=ANALYSIS_MODES,
            horizons=args.horizons,
            device=device,
            probe_epochs=args.probe_epochs,
            batch_size=args.batch_size,
            experiment=None,  # log complementarity separately below
            ds_tag=ds_tag,
        )

        for H in args.horizons:
            mse_t = ds_results.get("temporal_only", {}).get(H, {}).get("mse", float("nan"))
            mse_v = ds_results.get("visual_only", {}).get(H, {}).get("mse", float("nan"))
            mse_m = ds_results.get("multimodal", {}).get(H, {}).get("mse", float("nan"))

            try:
                complementarity = min(mse_t, mse_v) - mse_m
            except (TypeError, ValueError):
                complementarity = float("nan")

            rows.append({
                "dataset": ds_tag,
                "horizon": H,
                "mse_temporal": f"{mse_t:.4f}",
                "mse_visual":   f"{mse_v:.4f}",
                "mse_multimodal": f"{mse_m:.4f}",
                "complementarity": f"{complementarity:.4f}",
            })

            dominant = "temporal" if mse_t <= mse_v else "visual"
            c_sign = "+" if complementarity > 0 else ""
            print(
                f"  H={H:3d}  MSE_t={mse_t:.4f}  MSE_v={mse_v:.4f}  "
                f"MSE_mm={mse_m:.4f}  C={c_sign}{complementarity:.4f}  "
                f"[dominant={dominant}]"
            )

            if experiment is not None:
                experiment.log_metric(f"{ds_tag}/H{H}/mse_temporal", mse_t)
                experiment.log_metric(f"{ds_tag}/H{H}/mse_visual", mse_v)
                experiment.log_metric(f"{ds_tag}/H{H}/mse_multimodal", mse_m)
                experiment.log_metric(f"{ds_tag}/H{H}/complementarity", complementarity)

    print(f"\nTotal elapsed: {(time.time() - t_total) / 60:.1f} min")

    # ── save CSV ──────────────────────────────────────────────────────────────
    out_csv = args.results_dir / "ablation_G2_results.csv"
    fieldnames = ["dataset", "horizon", "mse_temporal", "mse_visual",
                  "mse_multimodal", "complementarity"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out_csv}")

    # ── summary ───────────────────────────────────────────────────────────────
    try:
        import statistics
        cs = [float(r["complementarity"]) for r in rows
              if r["complementarity"] not in ("nan", "")]
        if cs:
            pos = sum(1 for c in cs if c > 0)
            print(f"\nComplementarity summary:")
            print(f"  Mean C = {statistics.mean(cs):.4f}")
            print(f"  Fraction C > 0: {pos}/{len(cs)} ({100*pos/len(cs):.0f}%)")
            if pos / len(cs) >= 0.7:
                print("  → Branches are complementary on most (dataset, horizon) pairs.")
            else:
                print("  → One branch tends to dominate; fusion adds limited value.")
    except Exception:
        pass

    if experiment is not None:
        experiment.end()
        print("Comet ML experiment ended.")


if __name__ == "__main__":
    main()
