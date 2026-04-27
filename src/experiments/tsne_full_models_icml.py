"""
t-SNE comparison of all full-tier encoders on the 7 ICML benchmark datasets.

For each model (SimCLR, BYOL, CLIP, GRAM) the temporal encoder is used to
embed a fixed random sample from each dataset's test split.  t-SNE projects
the embeddings to 2-D.  A 2×2 panel figure is saved; one subplot per model.

Usage (GPU node):
    python src/experiments/tsne_full_models_icml.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for p in (SRC_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import training_utils as tu
from time_series_loader import TimeSeriesDataModule

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "SimCLR": {
        "config": SRC_DIR / "configs" / "lotsa_simclr_full.yaml",
        "ckpt_dir": ROOT_DIR / "checkpoints" / "simclr_full" / "ts_simclr_full_lotsa_20260414_002946",
    },
    "BYOL": {
        "config": SRC_DIR / "configs" / "lotsa_byol_bimodal_full.yaml",
        "ckpt_dir": ROOT_DIR / "checkpoints" / "byol_bimodal_full" / "ts_byol_bimodal_full_lotsa_20260414_171343",
    },
    "CLIP": {
        "config": SRC_DIR / "configs" / "lotsa_clip_full.yaml",
        "ckpt_dir": ROOT_DIR / "checkpoints" / "clip_full" / "ts_clip_full_lotsa_20260415_012856",
    },
    "GRAM": {
        "config": SRC_DIR / "configs" / "lotsa_gram_full.yaml",
        "ckpt_dir": ROOT_DIR / "checkpoints" / "gram_full" / "ts_gram_full_lotsa_20260412_121127",
    },
}

ICML_DATASETS = [
    ("ETTm1.csv",      "ETT-small"),
    ("ETTm2.csv",      "ETT-small"),
    ("ETTh1.csv",      "ETT-small"),
    ("ETTh2.csv",      "ETT-small"),
    ("weather.csv",    "weather"),
    ("traffic.csv",    "traffic"),
    ("electricity.csv","electricity"),
]

CONTEXT_LENGTH = 336
SAMPLES_PER_DS = 300
BATCH_SIZE = 128
TSNE_PERPLEXITY = 40
TSNE_ITERS = 1000

PALETTE = [
    "#e41a1c", "#e41a1c", "#e41a1c", "#e41a1c",   # 4 ETTs → red family
    "#4daf4a",   # weather → green
    "#377eb8",   # traffic → blue
    "#ff7f00",   # electricity → orange
]
LABELS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "weather", "traffic", "electricity"]
# Marker shapes to distinguish the 4 ETTs within the same colour
MARKERS = ["o", "s", "^", "D", "o", "o", "o"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dir(root: Path, ds_file: str) -> str:
    for p in root.rglob(ds_file):
        return str(p.parent)
    return str(root)


def _load_encoder(model_name: str, cfg_path: Path, ckpt_dir: Path,
                  device: torch.device) -> torch.nn.Module:
    config = tu.load_config(cfg_path)
    encoder = tu.build_encoder_from_config(config.model).to(device)

    for candidate in ("time_series_best.pt", "time_series_encoder.pt"):
        p = ckpt_dir / candidate
        if p.exists():
            state = torch.load(p, map_location=device)
            state = state.get("model_state_dict", state.get("model_state", state))
            encoder.load_state_dict(state)
            print(f"  [{model_name}] loaded encoder from {p.name}")
            break
    else:
        raise FileNotFoundError(f"No encoder checkpoint found in {ckpt_dir}")

    encoder.eval()
    return encoder


def _collect_embeddings(encoder: torch.nn.Module, icml_dir: Path,
                        device: torch.device) -> tuple[np.ndarray, list[str]]:
    """Return (N, D) embeddings and matching dataset label strings."""
    all_embs, all_labels = [], []

    for ds_file, ds_label in ICML_DATASETS:
        data_dir = _resolve_dir(icml_dir, ds_file)
        try:
            module = TimeSeriesDataModule(
                dataset_name=ds_file,
                data_dir=data_dir,
                batch_size=BATCH_SIZE,
                val_batch_size=BATCH_SIZE,
                num_workers=0,
                pin_memory=False,
                normalize=True,
                train=False, val=False, test=True,
                sample_size=(CONTEXT_LENGTH, 0, 1),
                scaler_type="standard",
            )
            module.setup()
            loader = module.test_loaders[0] if module.test_loaders else None
        except Exception as e:
            print(f"    [WARN] {ds_file}: {e}")
            continue

        if loader is None:
            continue

        collected = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].float().to(device)   # (B, L, C)
                B, L, C = x.shape
                # channel-independent: (B*C, 1, L)
                x_in = x.permute(0, 2, 1).reshape(B * C, 1, L)
                emb = encoder(x_in)
                if emb.dim() > 2:
                    emb = emb.flatten(1)
                collected.append(emb.cpu())
                if sum(t.shape[0] for t in collected) >= SAMPLES_PER_DS:
                    break

        if not collected:
            continue

        emb_cat = torch.cat(collected, dim=0)[:SAMPLES_PER_DS].numpy()
        all_embs.append(emb_cat)
        # use the short label (e.g. "ETTm1") not the folder name
        short = ds_file.replace(".csv", "")
        all_labels.extend([short] * emb_cat.shape[0])
        print(f"    {short}: {emb_cat.shape[0]} samples, dim={emb_cat.shape[1]}")

    if not all_embs:
        raise RuntimeError("No embeddings collected.")

    return np.concatenate(all_embs, axis=0), all_labels


def _fit_tsne(X: np.ndarray, seed: int = 42) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        learning_rate="auto",
        max_iter=TSNE_ITERS,
        random_state=seed,
        init="pca",
        verbose=0,
    )
    return tsne.fit_transform(X.astype(np.float32))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--icml_data_dir", type=Path,
                   default=ROOT_DIR / "ICML_datasets")
    p.add_argument("--results_dir", type=Path,
                   default=ROOT_DIR / "results" / "tsne_full_models")
    p.add_argument("--samples_per_ds", type=int, default=SAMPLES_PER_DS)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Unique dataset short names for legend
    ds_short = [f.replace(".csv", "") for f, _ in ICML_DATASETS]
    color_map = dict(zip(ds_short, PALETTE))
    marker_map = dict(zip(ds_short, MARKERS))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax_idx, (model_name, cfg) in enumerate(MODELS.items()):
        ax = axes[ax_idx]
        print(f"\n[{model_name}] Collecting embeddings...")

        try:
            encoder = _load_encoder(model_name, cfg["config"], cfg["ckpt_dir"], device)
        except Exception as e:
            ax.text(0.5, 0.5, f"Load error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(model_name)
            continue

        try:
            X, labels = _collect_embeddings(encoder, args.icml_data_dir, device)
        except Exception as e:
            ax.text(0.5, 0.5, f"Embed error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(model_name)
            continue

        print(f"  Total embeddings: {X.shape[0]} × {X.shape[1]} → fitting t-SNE...")
        coords = _fit_tsne(X, seed=args.seed)

        labels_arr = np.array(labels)
        unique_ds = ds_short  # preserve display order
        for ds, color, marker in zip(unique_ds, PALETTE, MARKERS):
            mask = labels_arr == ds
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, marker=marker,
                s=18, alpha=0.65, linewidths=0,
                label=ds,
            )

        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=7, markerscale=1.4, ncol=2, loc="best")

        # Save per-model CSV
        import pandas as pd
        df = pd.DataFrame({"dataset": labels, "x": coords[:, 0], "y": coords[:, 1]})
        df.to_csv(args.results_dir / f"tsne_{model_name.lower()}.csv", index=False)

    fig.suptitle(
        "t-SNE of Full-Tier Encoder Embeddings\n(7 ICML benchmark datasets, test split, "
        f"{args.samples_per_ds} samples/dataset)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = args.results_dir / "tsne_full_models_icml.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved panel figure → {out_path}")


if __name__ == "__main__":
    main()
