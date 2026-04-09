"""
Ablation G3 — Horizon × Branch Dominance
==========================================
Systematises the observation that the visual branch dominates at short horizons
(capturing local/periodic structure via Recurrence Plots) while the temporal
branch dominates at long horizons (SSM long-range dynamics).

Dominance metric (per dataset × horizon):
  dominant_branch = "temporal" if MSE_temporal ≤ MSE_visual else "visual"
  margin = |MSE_temporal − MSE_visual|

Two execution modes
-------------------
  Option B (fast): post-process an existing ablation_G_results.csv.
    Provide --g_results_csv path/to/ablation_G_results.csv

  Option A (fresh run): re-run linear probes using the same protocol as Ablation G.
    Provide --checkpoint_dir (or --random_weights for smoke tests).
    Only temporal_only and visual_only modes are probed (multimodal not needed).

Output
------
  ablation_G3_dominance.csv  — dataset × horizon rows:
    dataset, horizon, mse_temporal, mse_visual, dominant_branch, margin
  ablation_G3_heatmap.pdf   — dataset (rows) × horizon (cols) coloured by branch
    Blue  = temporal branch dominates
    Orange = visual branch dominates
    Cell text = margin value

Usage
-----
  # Option A — fresh run (smoke test with random weights)
  python src/experiments/ablation_G3_horizon_dominance.py \\
      --random_weights \\
      --config src/experiments/smoke_test/smoke_config.yaml \\
      --datasets ETTm1.csv \\
      --horizons 96 192 \\
      --probe_epochs 2 \\
      --results_dir results/smoke/ablation_G3 \\
      --no_comet

  # Option B — post-process existing ablation_G results
  python src/experiments/ablation_G3_horizon_dominance.py \\
      --g_results_csv results/ablation_G/ablation_G_results.csv \\
      --results_dir results/ablation_G3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu

# Reuse probe utilities from ablation_G — no code duplication
from ablation_G_encoder_modes import (
    probe_dataset,
    PROBE_DATASETS,
    HORIZONS,
)

import torch


# ── dominance analysis ────────────────────────────────────────────────────────

def _compute_dominance(
    rows_g: List[Dict],
) -> List[Dict]:
    """Given ablation_G-style rows, compute dominant branch per (dataset, horizon)."""
    table: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row in rows_g:
        key = (row["dataset"], int(row["horizon"]))
        table.setdefault(key, {})[row["mode"]] = float(row["mse"])

    rows_out = []
    for (ds, H), modes in sorted(table.items()):
        mse_t = modes.get("temporal_only", float("nan"))
        mse_v = modes.get("visual_only", float("nan"))
        try:
            dominant = "temporal" if mse_t <= mse_v else "visual"
            margin = abs(mse_t - mse_v)
        except (TypeError, ValueError):
            dominant = "unknown"
            margin = float("nan")
        rows_out.append({
            "dataset": ds,
            "horizon": H,
            "mse_temporal": f"{mse_t:.4f}",
            "mse_visual":   f"{mse_v:.4f}",
            "dominant_branch": dominant,
            "margin": f"{margin:.4f}",
        })
    return rows_out


# ── heatmap ───────────────────────────────────────────────────────────────────

def _plot_dominance_heatmap(rows_out: List[Dict], out_path: Path) -> None:
    """Dataset × horizon heatmap: blue=temporal dominant, orange=visual dominant."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors
        import numpy as np

        datasets = sorted({r["dataset"] for r in rows_out})
        horizons = sorted({int(r["horizon"]) for r in rows_out})

        Z = np.zeros((len(datasets), len(horizons)))
        M = np.zeros_like(Z)
        ds_idx = {d: i for i, d in enumerate(datasets)}
        h_idx  = {h: j for j, h in enumerate(horizons)}

        for r in rows_out:
            i = ds_idx[r["dataset"]]
            j = h_idx[int(r["horizon"])]
            Z[i, j] = 0.0 if r["dominant_branch"] == "temporal" else 1.0
            try:
                M[i, j] = float(r["margin"])
            except (ValueError, TypeError):
                M[i, j] = float("nan")

        cmap = matplotlib.colors.ListedColormap(["#4C72B0", "#DD8452"])  # blue / orange
        fig, ax = plt.subplots(figsize=(max(4, len(horizons) * 1.5),
                                        max(3, len(datasets) * 0.7)))
        ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        for i in range(len(datasets)):
            for j in range(len(horizons)):
                val = M[i, j]
                txt = f"{val:.3f}" if not (val != val) else "—"  # nan check
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"H={h}" for h in horizons])
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=8)
        ax.set_title(
            "Dominant branch per (dataset, horizon)\n"
            "Blue = temporal | Orange = visual   (cell text = |MSE_t − MSE_v|)"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Heatmap saved: {out_path}")
    except ImportError as e:
        print(f"  Warning: could not produce heatmap — {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Horizon × branch dominance analysis — Ablation G3"
    )

    # Option B — post-process existing ablation_G CSV
    p.add_argument("--g_results_csv", type=Path, default=None,
                   help="Path to an existing ablation_G_results.csv. When provided, "
                        "no fresh probing is done (Option B, much faster).")

    # Option A — fresh run arguments
    p.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="Directory with time_series_best.pt + visual_encoder_best.pt "
                        "(Option A: fresh run). Not required for Option B or --random_weights.")
    p.add_argument("--random_weights", action="store_true",
                   help="Use randomly initialised encoders (smoke test; Option A only).")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--probe_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--datasets", nargs="+", default=PROBE_DATASETS)
    p.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)

    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_G3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_comet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    rows_g: Optional[List[Dict]] = None

    # ── Option B: load existing ablation_G results ────────────────────────────
    if args.g_results_csv is not None:
        if not args.g_results_csv.exists():
            raise FileNotFoundError(f"--g_results_csv not found: {args.g_results_csv}")
        with args.g_results_csv.open() as f:
            rows_g = list(csv.DictReader(f))
        print(f"Loaded {len(rows_g)} rows from {args.g_results_csv}")

    else:
        # ── Option A: fresh probe run ─────────────────────────────────────────
        if not args.random_weights and args.checkpoint_dir is None:
            raise ValueError(
                "Provide --g_results_csv (Option B) or "
                "--checkpoint_dir / --random_weights (Option A)."
            )

        tu.set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        config = tu.load_config(args.config)

        encoder = tu.build_encoder_from_config(config.model).to(device)
        visual  = tu.build_visual_encoder_from_config(config.model).to(device)

        if args.random_weights:
            print("Using RANDOM weights (smoke-test mode — no checkpoint loaded).")
        else:
            enc_names = ("time_series_best.pt", "encoder.pt", "time_series_last.pt")
            vis_names = ("visual_encoder_best.pt", "visual_encoder.pt", "visual_encoder_last.pt")
            enc_ckpt = next(
                (args.checkpoint_dir / n for n in enc_names
                 if (args.checkpoint_dir / n).exists()),
                args.checkpoint_dir / "time_series_best.pt",
            )
            vis_ckpt = next(
                (args.checkpoint_dir / n for n in vis_names
                 if (args.checkpoint_dir / n).exists()),
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

        # Run probes for temporal_only and visual_only only (multimodal not needed)
        rows_g = []
        t_total = time.time()
        for ds in args.datasets:
            ds_tag = ds.replace(".csv", "").replace(".txt", "")
            print(f"\n{'='*60}\nDataset: {ds_tag}\n{'='*60}")

            ds_results = probe_dataset(
                encoder, visual,
                data_dir=args.data_dir,
                dataset_csv=ds,
                modes=["temporal_only", "visual_only"],
                horizons=args.horizons,
                device=device,
                probe_epochs=args.probe_epochs,
                batch_size=args.batch_size,
                experiment=None,
                ds_tag=ds_tag,
            )

            for mode, horizon_metrics in ds_results.items():
                for H, m in horizon_metrics.items():
                    rows_g.append({
                        "mode": mode,
                        "dataset": ds_tag,
                        "horizon": H,
                        "mse": f"{m['mse']:.4f}",
                        "mae": f"{m['mae']:.4f}",
                    })
                    print(f"  [{mode:15s}]  H={H:3d}  MSE={m['mse']:.4f}")

        print(f"\nProbing elapsed: {(time.time() - t_total) / 60:.1f} min")

    # ── Compute dominance ─────────────────────────────────────────────────────
    rows_out = _compute_dominance(rows_g)

    print("\nDominance results:")
    for r in rows_out:
        print(
            f"  {r['dataset']:20s}  H={r['horizon']:3d}  "
            f"MSE_t={r['mse_temporal']}  MSE_v={r['mse_visual']}  "
            f"dominant={r['dominant_branch']}  margin={r['margin']}"
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = args.results_dir / "ablation_G3_dominance.csv"
    fieldnames = ["dataset", "horizon", "mse_temporal", "mse_visual",
                  "dominant_branch", "margin"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nDominance CSV saved to {out_csv}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    _plot_dominance_heatmap(rows_out, args.results_dir / "ablation_G3_heatmap.pdf")

    # ── Summary ───────────────────────────────────────────────────────────────
    temporal_wins = sum(1 for r in rows_out if r["dominant_branch"] == "temporal")
    visual_wins   = sum(1 for r in rows_out if r["dominant_branch"] == "visual")
    total = len(rows_out)
    if total:
        print(
            f"\nSummary: temporal dominant in {temporal_wins}/{total} "
            f"({100*temporal_wins/total:.0f}%) of (dataset, horizon) pairs."
        )
        print(
            f"         visual   dominant in {visual_wins}/{total} "
            f"({100*visual_wins/total:.0f}%) of (dataset, horizon) pairs."
        )


if __name__ == "__main__":
    main()
