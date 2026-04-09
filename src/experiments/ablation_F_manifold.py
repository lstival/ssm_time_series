"""
Ablation F — Manifold Quality on Evaluation Data (refactored Table 4)
======================================================================
Original Table 4 used TRAINING data embeddings — reviewers flag this as
evidence of memorisation rather than transfer.

This script re-runs the manifold analysis on EVALUATION datasets that were
NEVER seen during pre-training, producing:
  • t-SNE plot coloured by dataset origin (PDF/PNG)
  • Silhouette score per encoder mode
  • Davies-Bouldin index per encoder mode
  • Cohesion / Separation ratio

Encoder modes compared (all using the SAME pretrained checkpoint):
  - temporal_only       : MambaEncoder embeddings only
  - visual_only         : MambaVisualEncoder embeddings only
  - multimodal          : concatenation [temporal ‖ visual]
  - random_temporal     : randomly initialised MambaEncoder (untrained lower bound)
  - random_visual       : randomly initialised MambaVisualEncoder (untrained lower bound)
  - random_multimodal   : concatenation of random encoders
  - supervised_*        : (optional) encoders loaded from --supervised_ckpt (upper bound)

Datasets (4 from Table 4 for comparability):
  Electricity, Solar, Weather, Exchange

If clusters discriminate on UNSEEN data → evidence of transfer.
If they collapse → fundamental limitation (discuss in Section 5 / Limitations).

Usage
-----
  # Full run with pretrained checkpoint
  python src/experiments/ablation_F_manifold.py \
      --checkpoint_dir checkpoints/ts_encoder_lotsa \
      --config src/configs/lotsa_clip.yaml \
      --data_dir ICML_datasets \
      --results_dir results/ablation_F

  # Smoke test (no checkpoint needed)
  python src/experiments/ablation_F_manifold.py \
      --random_weights \
      --config src/configs/lotsa_clip.yaml \
      --data_dir ICML_datasets \
      --results_dir results/smoke/ablation_F
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import training_utils as tu
from util import prepare_sequence, reshape_multivariate_series
from time_series_loader import TimeSeriesDataModule

EVAL_DATASETS: List[str] = [
    "electricity.csv",
    "solar_AL.txt",
    "weather.csv",
    "exchange_rate.csv",
]
ENCODER_MODES: List[str] = ["temporal_only", "visual_only", "multimodal"]
MAX_SAMPLES_PER_DS: int = 500   # cap for t-SNE feasibility


# ── embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def _extract_embeddings(
    encoder: nn.Module,
    visual: nn.Module,
    loader,
    device: torch.device,
    max_samples: int = MAX_SAMPLES_PER_DS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_temporal, z_visual, z_multi) arrays of shape (N, D)."""
    encoder.eval(); visual.eval()
    zts, zvs, zms = [], [], []
    for batch in loader:
        if len(zts) * 64 >= max_samples:
            break
        x = batch[0].to(device).float()
        x = reshape_multivariate_series(prepare_sequence(x))
        zt = encoder(x)
        zv = visual(x)
        zts.append(zt.cpu().numpy())
        zvs.append(zv.cpu().numpy())
        zms.append(torch.cat([zt, zv], dim=1).cpu().numpy())

    def _cat_limit(lst):
        arr = np.concatenate(lst, axis=0)
        return arr[:max_samples]

    return _cat_limit(zts), _cat_limit(zvs), _cat_limit(zms)


# ── clustering metrics ────────────────────────────────────────────────────────

def _compute_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    unique = np.unique(labels)
    if len(unique) < 2 or embeddings.shape[0] < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "cohesion": float("nan"), "separation": float("nan")}

    sil = silhouette_score(embeddings, labels, metric="euclidean")
    db = davies_bouldin_score(embeddings, labels)

    # Cohesion: mean intra-cluster distance; Separation: mean inter-cluster centroid distance
    centroids = {lbl: embeddings[labels == lbl].mean(0) for lbl in unique}
    intra = np.mean([
        np.mean(np.linalg.norm(embeddings[labels == lbl] - centroids[lbl], axis=1))
        for lbl in unique
    ])
    inter_dists = []
    for i, li in enumerate(unique):
        for lj in unique[i+1:]:
            inter_dists.append(np.linalg.norm(centroids[li] - centroids[lj]))
    inter = np.mean(inter_dists) if inter_dists else float("nan")

    return {"silhouette": float(sil), "davies_bouldin": float(db),
            "cohesion": float(intra), "separation": float(inter)}


# ── t-SNE visualisation ───────────────────────────────────────────────────────

def _plot_tsne(embeddings: np.ndarray, labels: np.ndarray, ds_names: List[str],
               title: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        proj = tsne.fit_transform(embeddings)

        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("tab10", len(unique_labels))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            name = ds_names[lbl] if lbl < len(ds_names) else str(lbl)
            ax.scatter(proj[mask, 0], proj[mask, 1], c=[colors(i)], label=name,
                       s=10, alpha=0.7)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except ImportError as e:
        print(f"  Warning: could not produce t-SNE plot — {e}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="Directory containing time_series_best.pt + visual_encoder_best.pt. "
                        "Not required when --random_weights is set.")
    p.add_argument("--random_weights", action="store_true",
                   help="Use randomly initialised encoders for the primary modes "
                        "(smoke-test mode; no checkpoint needed).")
    p.add_argument("--random_baseline", action="store_true", default=True,
                   help="Also compute random encoder baselines (default: True). "
                        "Disable with --no_random_baseline.")
    p.add_argument("--no_random_baseline", dest="random_baseline", action="store_false")
    p.add_argument("--supervised_ckpt", type=Path, default=None,
                   help="(Optional) directory with a supervised checkpoint. Adds "
                        "supervised_temporal/visual/multimodal rows. Skipped with a "
                        "warning if the path does not exist.")
    p.add_argument("--config", type=Path,
                   default=src_dir / "configs" / "lotsa_clip.yaml")
    p.add_argument("--data_dir", type=Path,
                   default=src_dir.parent / "ICML_datasets")
    p.add_argument("--results_dir", type=Path,
                   default=src_dir.parent / "results" / "ablation_F")
    p.add_argument("--datasets", nargs="+", default=EVAL_DATASETS)
    p.add_argument("--max_samples", type=int, default=MAX_SAMPLES_PER_DS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _get_loader(ds_csv: str, data_dir: Path, batch_size: int = 64):
    """Build and return a data loader for a dataset, preferring test split."""
    module = TimeSeriesDataModule(
        dataset_name=ds_csv,
        data_dir=str(data_dir),
        batch_size=batch_size, val_batch_size=batch_size,
        num_workers=0, pin_memory=False,
        normalize=True, train=False, val=False, test=True,
    )
    module.setup()
    test_loaders = module.test_loaders
    train_loaders = module.train_loaders
    if not test_loaders and not train_loaders:
        return None
    return (test_loaders or train_loaders)[0]


def main() -> None:
    args = parse_args()

    if not args.random_weights and args.checkpoint_dir is None:
        raise ValueError("Provide --checkpoint_dir or use --random_weights for smoke testing")

    tu.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = tu.load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # ── load pretrained encoders ─────────────────────────────────────────────
    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual = tu.build_visual_encoder_from_config(config.model).to(device)

    def _load(module, path):
        state = torch.load(path, map_location="cpu")
        key = "model_state_dict" if "model_state_dict" in state else "model_state"
        module.load_state_dict(state[key])
        print(f"  Loaded {path.name}")

    if args.random_weights:
        print("Using RANDOM weights for primary encoders (smoke-test mode).")
    else:
        # Accept both LOTSA training names and smoke-test names saved by ablation A
        ckpt_ts = next(
            (args.checkpoint_dir / n for n in ("time_series_best.pt", "encoder.pt")
             if (args.checkpoint_dir / n).exists()),
            args.checkpoint_dir / "time_series_best.pt",
        )
        ckpt_vis = next(
            (args.checkpoint_dir / n for n in ("visual_encoder_best.pt", "visual_encoder.pt")
             if (args.checkpoint_dir / n).exists()),
            args.checkpoint_dir / "visual_encoder_best.pt",
        )
        if not ckpt_ts.exists():
            raise FileNotFoundError(f"Temporal encoder checkpoint not found: {ckpt_ts}")
        if not ckpt_vis.exists():
            raise FileNotFoundError(f"Visual encoder checkpoint not found: {ckpt_vis}")
        _load(encoder, ckpt_ts)
        _load(visual, ckpt_vis)

    # ── collect embeddings per dataset ───────────────────────────────────────
    all_zt, all_zv, all_zm = [], [], []
    all_labels = []
    ds_names = []

    for ds_idx, ds_csv in enumerate(args.datasets):
        ds_name = ds_csv.replace(".csv", "").replace(".txt", "")
        loader = _get_loader(ds_csv, args.data_dir)
        if loader is None:
            print(f"  Warning: no loader for {ds_csv}, skipping")
            continue

        zt, zv, zm = _extract_embeddings(
            encoder, visual, loader, device, max_samples=args.max_samples
        )
        n = zt.shape[0]
        all_zt.append(zt); all_zv.append(zv); all_zm.append(zm)
        all_labels.append(np.full(n, ds_idx, dtype=np.int32))
        ds_names.append(ds_name)
        print(f"  {ds_name}: {n} embeddings collected")

    if not all_zt:
        raise RuntimeError("No embeddings collected — check --data_dir and --datasets.")

    zt_all = np.concatenate(all_zt, axis=0)
    zv_all = np.concatenate(all_zv, axis=0)
    zm_all = np.concatenate(all_zm, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # ── compute metrics and t-SNE per encoder mode ───────────────────────────
    mode_embs = {
        "temporal_only": zt_all,
        "visual_only":   zv_all,
        "multimodal":    zm_all,
    }

    # ── random encoder baselines ──────────────────────────────────────────────
    if args.random_baseline:
        print("\nBuilding random encoder baselines (untrained weights)...")
        rand_enc = tu.build_encoder_from_config(config.model).to(device)
        rand_vis = tu.build_visual_encoder_from_config(config.model).to(device)
        # Deliberately NOT loading any checkpoint — weights stay random.
        rand_zt_parts, rand_zv_parts, rand_zm_parts = [], [], []
        for ds_idx, ds_csv in enumerate(args.datasets):
            ds_name = ds_csv.replace(".csv", "").replace(".txt", "")
            if ds_name not in ds_names:
                continue  # skip datasets that were unavailable in the main loop
            loader = _get_loader(ds_csv, args.data_dir)
            if loader is None:
                continue
            rzt, rzv, rzm = _extract_embeddings(
                rand_enc, rand_vis, loader, device, max_samples=args.max_samples
            )
            rand_zt_parts.append(rzt)
            rand_zv_parts.append(rzv)
            rand_zm_parts.append(rzm)
            print(f"  {ds_name}: {rzt.shape[0]} random embeddings collected")

        if rand_zt_parts:
            mode_embs["random_temporal"]   = np.concatenate(rand_zt_parts, axis=0)
            mode_embs["random_visual"]     = np.concatenate(rand_zv_parts, axis=0)
            mode_embs["random_multimodal"] = np.concatenate(rand_zm_parts, axis=0)

    # ── optional supervised encoder baseline ─────────────────────────────────
    if args.supervised_ckpt is not None:
        sup_ckpt = Path(args.supervised_ckpt)
        if not sup_ckpt.exists():
            print(f"[WARN] --supervised_ckpt {sup_ckpt} not found; skipping supervised baseline.")
        else:
            print(f"\nLoading supervised checkpoint from {sup_ckpt} ...")
            sup_enc = tu.build_encoder_from_config(config.model).to(device)
            sup_vis = tu.build_visual_encoder_from_config(config.model).to(device)
            sup_ckpt_ts = next(
                (sup_ckpt / n for n in ("time_series_best.pt", "encoder.pt")
                 if (sup_ckpt / n).exists()),
                sup_ckpt / "time_series_best.pt",
            )
            sup_ckpt_vis = next(
                (sup_ckpt / n for n in ("visual_encoder_best.pt", "visual_encoder.pt")
                 if (sup_ckpt / n).exists()),
                sup_ckpt / "visual_encoder_best.pt",
            )
            _load(sup_enc, sup_ckpt_ts)
            _load(sup_vis, sup_ckpt_vis)
            sup_zt_parts, sup_zv_parts, sup_zm_parts = [], [], []
            for ds_idx, ds_csv in enumerate(args.datasets):
                ds_name = ds_csv.replace(".csv", "").replace(".txt", "")
                if ds_name not in ds_names:
                    continue
                loader = _get_loader(ds_csv, args.data_dir)
                if loader is None:
                    continue
                szt, szv, szm = _extract_embeddings(
                    sup_enc, sup_vis, loader, device, max_samples=args.max_samples
                )
                sup_zt_parts.append(szt)
                sup_zv_parts.append(szv)
                sup_zm_parts.append(szm)
            if sup_zt_parts:
                mode_embs["supervised_temporal"]   = np.concatenate(sup_zt_parts, axis=0)
                mode_embs["supervised_visual"]     = np.concatenate(sup_zv_parts, axis=0)
                mode_embs["supervised_multimodal"] = np.concatenate(sup_zm_parts, axis=0)

    rows: List[Dict] = []
    for mode, embs in mode_embs.items():
        print(f"\nMode: {mode}")
        metrics = _compute_metrics(embs, labels)
        for k, v in metrics.items():
            print(f"  {k:18s}: {v:.4f}")
        rows.append({
            "mode": mode,
            "silhouette": f"{metrics['silhouette']:.4f}",
            "davies_bouldin": f"{metrics['davies_bouldin']:.4f}",
            "cohesion": f"{metrics['cohesion']:.4f}",
            "separation": f"{metrics['separation']:.4f}",
        })
        _plot_tsne(
            embs, labels, ds_names,
            title=f"t-SNE — {mode} (unseen data)",
            out_path=args.results_dir / f"tsne_{mode}.pdf",
        )

    # ── save metrics CSV ──────────────────────────────────────────────────────
    out_csv = args.results_dir / "ablation_F_metrics.csv"
    fieldnames = ["mode", "silhouette", "davies_bouldin", "cohesion", "separation"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    print(f"\nMetrics saved to {out_csv}")

    # ── interpretation ────────────────────────────────────────────────────────
    try:
        best_sil = max(rows, key=lambda r: float(r["silhouette"]))
        print(f"\nHighest Silhouette: {best_sil['mode']} ({best_sil['silhouette']})")
        print("Interpretation:")
        sil_val = float(best_sil["silhouette"])
        if sil_val > 0.3:
            print("  → Clusters discriminate on unseen data → evidence of transfer learning.")
        elif sil_val > 0.0:
            print("  → Weak cluster structure → limited but present transfer.")
        else:
            print("  → Embeddings collapse → discuss as limitation in paper.")
    except (ValueError, IndexError):
        pass


if __name__ == "__main__":
    main()
