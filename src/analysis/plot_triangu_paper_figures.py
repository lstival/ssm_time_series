#!/usr/bin/env python3
"""Produce individual paper figures for the TriangU upper-triangle RP method.

Outputs (all in results/triangu_paper/):
  fig_raw_series.pdf/png       – raw time series with 3 coloured patch spans
  fig_patch_1.pdf/png          – patch 1 values
  fig_patch_2.pdf/png          – patch 2 values
  fig_patch_3.pdf/png          – patch 3 values
  fig_rp_1.pdf/png             – RP of patch 1, light colourmap in patch colour
  fig_rp_2.pdf/png             – RP of patch 2
  fig_rp_3.pdf/png             – RP of patch 3
  fig_lag_token_1.pdf/png      – lag-token matrix for patch 1
  fig_lag_token_2.pdf/png      – lag-token matrix for patch 2
  fig_lag_token_3.pdf/png      – lag-token matrix for patch 3

Lag-token matrix colour rules:
  - Valid upper-triangle values rendered with a light two-colour palette
    blending from white to the patch colour.
  - Unused (padded) positions overlaid with a semi-transparent red (gamma ~0.45)
    so the structure underneath is still visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Patch colours (same as the original figure)
# ---------------------------------------------------------------------------
PATCH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue, orange, green
PATCH_ALPHA_SPAN = 0.22                              # transparency of span in raw series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_rp(x: np.ndarray) -> np.ndarray:
    """RP: abs diff, max-normalised (same as UpperTriDiagRPEncoder)."""
    rp = np.abs(x[:, None] - x[None, :])
    mx = float(np.max(rp))
    if mx > 1e-8:
        rp = rp / mx
    return rp


def triang_u_tokens(rp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Extract padded lag-token matrix from an RP.

    Returns:
      tokens  (L-1, L-1)  – padded values (zeros in unused positions)
      mask    (L-1, L-1)  – True where valid
      lengths             – number of valid elements per lag token
    """
    l = rp.shape[0]
    max_len = l - 1
    tokens = np.zeros((l - 1, max_len), dtype=np.float32)
    mask = np.zeros((l - 1, max_len), dtype=bool)
    lengths: List[int] = []

    for lag in range(1, l):
        rows = np.arange(l - lag)
        cols = rows + lag
        vals = rp[rows, cols]
        n = len(vals)
        tokens[lag - 1, :n] = vals
        mask[lag - 1, :n] = True
        lengths.append(n)

    return tokens, mask, lengths


def light_cmap_for_color(hex_color: str) -> mcolors.LinearSegmentedColormap:
    """Create a colourmap from near-white to the given hex colour."""
    rgb = mcolors.to_rgb(hex_color)
    # start slightly off-white so the low end is still visible on white paper
    start = tuple(min(1.0, c + 0.55 * (1.0 - c)) for c in rgb)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f"light_{hex_color}", [start, rgb], N=256
    )
    return cmap


def save(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 200) -> None:
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Scale all text 3× relative to matplotlib defaults
    base = 10
    S = 10  # text scale factor
    plt.rcParams.update({
        "font.size":         base * S,
        "axes.titlesize":    base * S,
        "axes.labelsize":    base * S,
        "xtick.labelsize":   base * S,
        "ytick.labelsize":   base * S,
        "legend.fontsize":   base * S,
        "figure.titlesize":  base * S,
    })

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results" / "triangu_paper_10x"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ---------------------------------------------------------
    csv_path = root / "ICML_datasets" / "ETT-small" / "ETTh1.csv"
    df = pd.read_csv(csv_path)
    col = "OT" if "OT" in df.columns else next(
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    )
    series = df[col].to_numpy(dtype=np.float32)

    patch_len = 64
    n_patches = 3
    start = 0
    end = start + patch_len * n_patches

    seg = series[start:end]
    patches = [seg[i * patch_len:(i + 1) * patch_len] for i in range(n_patches)]
    rps = [compute_rp(p) for p in patches]
    lag_data = [triang_u_tokens(rp) for rp in rps]

    # ---- Figure 1: raw time series ----------------------------------------
    fig, ax = plt.subplots(figsize=(6 * S, 2.8 * S))
    x = np.arange(len(seg))
    ax.plot(x, seg, lw=1.4, color="black", zorder=3)
    for i in range(n_patches):
        s = i * patch_len
        e = (i + 1) * patch_len
        ax.axvspan(s, e - 1, color=PATCH_COLORS[i], alpha=PATCH_ALPHA_SPAN, zorder=2)
    ax.set_xlabel("Time index")
    ax.set_ylabel(col)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, out_dir, "fig_raw_series")
    print("Saved fig_raw_series")

    # ---- Figures 2-4: individual patch values ------------------------------
    for i, (patch, color) in enumerate(zip(patches, PATCH_COLORS)):
        fig, ax = plt.subplots(figsize=(4 * S, 2.6 * S))
        ax.plot(np.arange(patch_len), patch, lw=1.6, color=color)
        ax.set_xlabel("Within-patch index")
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, out_dir, f"fig_patch_{i+1}")
        print(f"Saved fig_patch_{i+1}")

    # ---- Figures 5-7: recurrence plots (light colour palettes) -------------
    for i, (rp, color) in enumerate(zip(rps, PATCH_COLORS)):
        cmap = light_cmap_for_color(color)
        fig, ax = plt.subplots(figsize=(4 * S, 3.6 * S))
        im = ax.imshow(rp, cmap=cmap, aspect="equal", vmin=0, vmax=1,
                       origin="upper")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Normalised |x_i − x_j|")
        ax.set_xlabel("j")
        ax.set_ylabel("i")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, out_dir, f"fig_rp_{i+1}")
        print(f"Saved fig_rp_{i+1}")

    # ---- Figures 8-10: lag-token matrices (one per RP) --------------------
    # Layout: valid values in patch-coloured light cmap;
    #         unused (padded) positions overlaid with semi-transparent red.
    RED_OVERLAY = np.array([1.0, 0.0, 0.0, 0.45])  # RGBA; alpha = gamma ~0.45

    for i, ((tokens, mask, lengths), color) in enumerate(zip(lag_data, PATCH_COLORS)):
        cmap = light_cmap_for_color(color)

        # Align valid values to the upper-right corner (flip left-right so
        # lag-1 token occupies the right end — matches RP upper-triangle
        # intuition: short lags are near the diagonal ≈ right of each row).
        tokens_display = np.fliplr(tokens)
        mask_display = np.fliplr(mask)

        # Build RGBA image manually so we can overlay the red mask exactly.
        norm = mcolors.Normalize(vmin=0, vmax=1)
        rgba = cmap(norm(tokens_display))           # (H, W, 4)

        # Where mask is invalid: set pixel to white first, then red overlay
        invalid = ~mask_display                     # (H, W)
        rgba[invalid] = [1.0, 1.0, 1.0, 1.0]       # white base for invalid

        # Composite red on top with alpha = 0.45 (gamma-like transparency)
        alpha_r = RED_OVERLAY[3]
        rgba[invalid, 0] = alpha_r * RED_OVERLAY[0] + (1 - alpha_r) * rgba[invalid, 0]
        rgba[invalid, 1] = alpha_r * RED_OVERLAY[1] + (1 - alpha_r) * rgba[invalid, 1]
        rgba[invalid, 2] = alpha_r * RED_OVERLAY[2] + (1 - alpha_r) * rgba[invalid, 2]

        fig, ax = plt.subplots(figsize=(5 * S, 4 * S))
        ax.imshow(rgba, aspect="auto", origin="upper")

        # Add a scalar mappable colourbar for the valid-value range
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("RP value (valid)")

        # Add a small red patch to legend explaining the overlay
        red_patch = mpl.patches.Patch(facecolor=(1, 0.55, 0.55), label="Unused (padded)")
        ax.legend(handles=[red_patch], loc="lower left",
                  framealpha=0.85, edgecolor="none")

        ax.set_xlabel("Token element index (max L−1)")
        ax.set_ylabel("Lag k (1 → L−1, top to bottom)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save(fig, out_dir, f"fig_lag_token_{i+1}")
        print(f"Saved fig_lag_token_{i+1}")

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
