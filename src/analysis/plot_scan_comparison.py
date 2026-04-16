#!/usr/bin/env python3
"""Figure: scan-path comparison on a small RP grid.

Two side-by-side panels on a small L=8 RP:
  Left  — Normal raster scan (row by row, left→right), like VMamba route 1.
  Right — UpperTriDiag lag-token scan (anti-diagonal by anti-diagonal,
           reading lag-1 … lag-L-1). Each lag shown in a distinct colour.

Colour palette: Wong (2011) 8-colour set — safe for deuteranopia, tritanopia,
and greyscale printing.  Arrow colours are also from this set.

Legends and annotations are placed OUTSIDE the axes.

Output: results/triangu_paper/fig_scan_comparison.pdf / .png
"""

from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# ── style ──────────────────────────────────────────────────────────────────────
FONT = 14 * 10   # 10× base size
mpl.rcParams.update({
    "font.size":       FONT,
    "axes.labelsize":  FONT,
    "xtick.labelsize": FONT - 20,
    "ytick.labelsize": FONT - 20,
    "axes.titlesize":  FONT,
    "legend.fontsize": FONT - 10,
    "legend.title_fontsize": FONT - 10,
})

L = 8   # grid size

# ── Wong (2011) colorblind-safe palette ────────────────────────────────────────
# Works in deuteranopia, protanopia, tritanopia AND greyscale.
# Ordered so that adjacent lags have maximally different luminance too.
WONG = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black  (lag 8 — only needed if L>8)
]

# Raster-scan arrow colours (also Wong-safe)
COL_SCAN   = "#0072B2"   # blue  — scan direction
COL_RETURN = "#999999"   # grey  — return jump (readable in greyscale)
COL_START  = "#009E73"   # bluish green
COL_END    = "#D55E00"   # vermillion

# Lower-tri unused region — use a neutral hatch pattern instead of colour fill
# so it reads clearly in greyscale
LOWER_TRI_COLOR = "#BBBBBB"
LOWER_TRI_ALPHA = 0.35


# ── helpers ────────────────────────────────────────────────────────────────────

def cell_centre(r: int, c: int) -> tuple[float, float]:
    return c + 0.5, -(r + 0.5)


def draw_rp_grid(ax, rp: np.ndarray, alpha: float = 0.15) -> None:
    ax.imshow(rp, cmap="Greys", vmin=0, vmax=1, alpha=alpha,
              extent=[0, L, -L, 0], aspect="equal", origin="upper")
    for k in range(L + 1):
        ax.axhline(-k, color="#AAAAAA", lw=4.0, zorder=0)
        ax.axvline( k, color="#AAAAAA", lw=4.0, zorder=0)
    ax.set_xlim(0, L);  ax.set_ylim(-L, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(L) + 0.5);  ax.set_xticklabels(range(L))
    ax.set_yticks(-(np.arange(L) + 0.5));  ax.set_yticklabels(range(L))
    ax.set_xlabel("j  (column)")
    ax.set_ylabel("i  (row)")


def ann_arrow(ax, src, dst, color, lw=12.0, rad=0.0, dashed=False, ms=60):
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    mutation_scale=ms, lw=lw,
                    linestyle="dashed" if dashed else "solid",
                    connectionstyle=f"arc3,rad={rad}"),
                zorder=4)


def dot(ax, pos, color, size=2000, marker="o"):
    ax.scatter(*pos, s=size, color=color, marker=marker,
               zorder=6, linewidths=3.0, edgecolors="white")


def make_rp(L: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.normal(size=L)).astype(np.float32)
    rp = np.abs(x[:, None] - x[None, :])
    rp /= rp.max() + 1e-8
    return rp


# ── Panel A: raster scan ──────────────────────────────────────────────────────

def draw_raster(ax, rp: np.ndarray) -> list:
    """Returns legend handles (to be placed outside by the caller)."""
    draw_rp_grid(ax, rp)
    ax.set_title("Normal raster scan\n(row-by-row, all cells)", pad=8)

    # Cell gradient: light→saturated using Wong orange (#E69F00) → blue (#0072B2)
    # Distinct from grey (unused) and black arrows; readable in greyscale by luminance
    seq = [(r, c) for r in range(L) for c in range(L)]
    n = len(seq)
    cmap_raster = mcolors.LinearSegmentedColormap.from_list(
        "raster", ["#FFF5CC", "#E69F00", "#0072B2"], N=256)
    for idx, (r, c) in enumerate(seq):
        col = cmap_raster(idx / (n - 1))
        rect = mpatches.FancyBboxPatch(
            (c + 0.08, -(r + 0.92)), 0.84, 0.84,
            boxstyle="round,pad=0.04",
            linewidth=0, facecolor=mcolors.to_rgba(col, 0.65), zorder=1)
        ax.add_patch(rect)

    # Arrows row by row
    for r in range(L):
        ann_arrow(ax, cell_centre(r, 0), cell_centre(r, L - 1),
                  color=COL_SCAN, lw=12.0, rad=0.0)
        if r < L - 1:
            ann_arrow(ax, cell_centre(r, L - 1), cell_centre(r + 1, 0),
                      color=COL_RETURN, lw=6.0, rad=0.45, dashed=True, ms=40)

    dot(ax, cell_centre(0, 0),     COL_START, marker="o")
    dot(ax, cell_centre(L-1, L-1), COL_END,   marker="s")

    handles = [
        mpl.lines.Line2D([0], [0], color=COL_SCAN,  lw=8, label="scan direction"),
        mpl.lines.Line2D([0], [0], color=COL_RETURN, lw=5, ls="--", label="return jump"),
        mpl.lines.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=COL_START, markersize=24, label="start"),
        mpl.lines.Line2D([0], [0], marker="s", color="w",
                         markerfacecolor=COL_END,   markersize=24, label="end"),
    ]
    return handles


# ── Panel B: upper-tri lag-token scan ─────────────────────────────────────────

def draw_upper_tri(ax, rp: np.ndarray) -> list:
    """Returns legend handles (to be placed outside by the caller)."""
    draw_rp_grid(ax, rp)
    ax.set_title("UpperTriDiag scan\n(anti-diagonal lag tokens)", pad=8)

    n_lags = L - 1
    lag_colors = [WONG[k % len(WONG)] for k in range(n_lags)]

    legend_handles = []

    for k in range(1, L):
        color = lag_colors[k - 1]
        rows = list(range(L - k))
        cols = [r + k for r in rows]

        for r, c in zip(rows, cols):
            rect = mpatches.FancyBboxPatch(
                (c + 0.08, -(r + 0.92)), 0.84, 0.84,
                boxstyle="round,pad=0.04",
                linewidth=1.4, edgecolor=color,
                facecolor=mcolors.to_rgba(color, 0.40),
                zorder=2)
            ax.add_patch(rect)

        if len(rows) >= 2:
            for step in range(len(rows) - 1):
                ann_arrow(ax,
                          cell_centre(rows[step],     cols[step]),
                          cell_centre(rows[step + 1], cols[step + 1]),
                          color=color, lw=12.0, rad=0.0, ms=55)
            dot(ax, cell_centre(rows[0], cols[0]), color=color, size=2000)
        elif len(rows) == 1:
            dot(ax, cell_centre(rows[0], cols[0]), color=color, size=2200)

        if k < L - 1:
            ann_arrow(ax,
                      cell_centre(rows[-1], cols[-1]),
                      cell_centre(0, k + 1),
                      color=color, lw=5.0, rad=-0.4, dashed=True, ms=35)

        legend_handles.append(
            mpatches.Patch(facecolor=mcolors.to_rgba(color, 0.6),
                           edgecolor=color, label=f"lag {k}"))

    # Lower triangle: grey hatched (greyscale-safe, no reliance on colour alone)
    lower = np.zeros((L, L))
    for r in range(L):
        for c in range(r):
            lower[r, c] = 1.0
    ax.imshow(lower,
              cmap=mcolors.LinearSegmentedColormap.from_list(
                  "lt", [(1, 1, 1, 0), (0.7, 0.7, 0.7, LOWER_TRI_ALPHA)]),
              extent=[0, L, -L, 0], aspect="equal", origin="upper", zorder=1)

    # Diagonal marker
    ax.plot([c + 0.5 for c in range(L)],
            [-(r + 0.5) for r in range(L)],
            color="#555555", lw=6.0, ls=":", zorder=3)

    legend_handles.append(
        mpatches.Patch(facecolor=(0.7, 0.7, 0.7, LOWER_TRI_ALPHA),
                       edgecolor="#888888", label="unused (lower tri)"))
    return legend_handles


# ── main ───────────────────────────────────────────────────────────────────────

def save(fig, out_dir, stem, dpi=200):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


FIG_W, FIG_H = 80, 90   # large canvas so 10× font fits comfortably


def make_fig_raster(rp, out_dir):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    handles = draw_raster(ax, rp)
    # reserve 15% at bottom for legend only
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(handles=handles,
               loc="lower left", bbox_to_anchor=(0.02, 0.02),
               ncol=4, framealpha=0.9, edgecolor="none")
    save(fig, out_dir, "fig_scan_raster", dpi=20)
    print(f"  saved fig_scan_raster")


def make_fig_upper_tri(rp, out_dir):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    handles = draw_upper_tri(ax, rp)
    # reserve 22% at bottom: ~12% legend + ~7% caption + 3% padding
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    fig.legend(handles=handles,
               loc="lower center", bbox_to_anchor=(0.5, 0.10),
               ncol=4, framealpha=0.9, edgecolor="none")
    fig.text(0.5, 0.03,
             "← anti-diagonals extracted in lag order (lag 1 → lag L−1)",
             ha="center", va="bottom", style="italic", color="#444444")
    save(fig, out_dir, "fig_scan_upper_tri", dpi=20)
    print(f"  saved fig_scan_upper_tri")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results" / "triangu_paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    rp = make_rp(L)
    make_fig_raster(rp, out_dir)
    make_fig_upper_tri(rp, out_dir)


if __name__ == "__main__":
    main()
