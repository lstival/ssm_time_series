#!/usr/bin/env python3
"""Scan-step figures for paper.

Normal SS2D (4 scans) — 4 individual panels:
  A  row-major forward        →  (left-to-right, top-to-bottom)
  B  col-major forward        ↓  (top-to-bottom, left-to-right)
  C  row-major backward       ←  (right-to-left, bottom-to-top)
  D  col-major backward       ↑  (bottom-to-top, right-to-left)

UpperTriDiag — 2 individual panels:
  1  Upper-triangle extraction: anti-diagonals read as lag tokens
     (shows which cells are read and in what colour/order, lower-tri masked red)
  2  Mamba sequence scan: the flattened lag-token sequence fed left-to-right
     into Mamba (token 1 = lag-1 anti-diag … token L-1 = lag-(L-1) anti-diag)

All saved as PDF+PNG in results/triangu_paper/.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# ── global style ───────────────────────────────────────────────────────────────
FONT = 15
mpl.rcParams.update({
    "font.size":       FONT,
    "axes.labelsize":  FONT,
    "axes.titlesize":  FONT + 1,
    "xtick.labelsize": FONT - 2,
    "ytick.labelsize": FONT - 2,
    "legend.fontsize": FONT - 2,
})

L  = 8   # grid dimension (small so arrows are readable)
FIG_SZ = (6.5, 6.5)   # each individual panel

# ── colour constants ───────────────────────────────────────────────────────────
COL_A = "#c0392b"   # red    – scan A / row forward
COL_B = "#2980b9"   # blue   – scan B / col forward
COL_C = "#27ae60"   # green  – scan C / row backward
COL_D = "#8e44ad"   # purple – scan D / col backward
SCAN_COLORS = [COL_A, COL_B, COL_C, COL_D]

LAG_CMAP = plt.cm.tab10

# ── helpers ────────────────────────────────────────────────────────────────────

def make_rp(L: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.normal(size=L)).astype(np.float32)
    rp = np.abs(x[:, None] - x[None, :])
    rp /= rp.max() + 1e-8
    return rp


def cell_xy(r: int, c: int) -> tuple[float, float]:
    """Centre of cell (row r, col c); x = col, y = -row (row-0 at top)."""
    return c + 0.5, -(r + 0.5)


def draw_grid_bg(ax, rp: np.ndarray, alpha: float = 0.15) -> None:
    ax.imshow(rp, cmap="Blues", vmin=0, vmax=1, alpha=alpha,
              extent=[0, L, -L, 0], aspect="equal", origin="upper")
    for k in range(L + 1):
        ax.axhline(-k, color="grey", lw=0.45, zorder=0)
        ax.axvline( k, color="grey", lw=0.45, zorder=0)
    ax.set_xlim(0, L);  ax.set_ylim(-L, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(L) + 0.5);  ax.set_xticklabels(range(L))
    ax.set_yticks(-(np.arange(L) + 0.5));  ax.set_yticklabels(range(L))
    ax.set_xlabel("j  (column)");  ax.set_ylabel("i  (row)")


def ann_arrow(ax, src, dst, color, lw=2.2, rad=0.0, dashed=False, ms=14):
    ls = "dashed" if dashed else "solid"
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle="-|>", color=color,
                                mutation_scale=ms, lw=lw,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)


def dot(ax, pos, color, size=55):
    ax.scatter(*pos, s=size, color=color, zorder=6)


def colour_cells(ax, seq, cmap, n_total):
    """Fill cells with a gradient colour along the scan order."""
    for idx, (r, c) in enumerate(seq):
        col = cmap(idx / max(n_total - 1, 1))
        rect = mpatches.FancyBboxPatch(
            (c + 0.07, -(r + 0.93)), 0.86, 0.86,
            boxstyle="round,pad=0.03",
            linewidth=0, facecolor=mcolors.to_rgba(col, 0.50), zorder=1)
        ax.add_patch(rect)


def save(fig, out_dir: Path, stem: str, dpi: int = 200) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  SS2D — 4 scan panels
# ═══════════════════════════════════════════════════════════════════════════════

def _ss2d_panel(ax, rp, scan_id: int) -> None:
    """Draw one of the 4 SS2D scan passes onto ax."""
    draw_grid_bg(ax, rp)

    titles = [
        "Scan A — row-major forward  →",
        "Scan B — col-major forward  ↓",
        "Scan C — row-major backward ←",
        "Scan D — col-major backward ↑",
    ]
    color = SCAN_COLORS[scan_id]
    ax.set_title(titles[scan_id], pad=8)

    # Build the ordered scan sequence
    if scan_id == 0:   # A: row-major forward
        seq = [(r, c) for r in range(L) for c in range(L)]
    elif scan_id == 1: # B: col-major forward
        seq = [(r, c) for c in range(L) for r in range(L)]
    elif scan_id == 2: # C: row-major backward
        seq = [(r, c) for r in range(L - 1, -1, -1) for c in range(L - 1, -1, -1)]
    else:              # D: col-major backward
        seq = [(r, c) for c in range(L - 1, -1, -1) for r in range(L - 1, -1, -1)]

    colour_cells(ax, seq, plt.cm.plasma, len(seq))

    # Determine the "primary" stride direction and "line reset" direction
    # For A/C: each line is a row; for B/D: each line is a column
    if scan_id == 0:   # rows, forward
        lines = [[(r, c) for c in range(L)] for r in range(L)]
        next_start = lambda r: (r + 1, 0) if r + 1 < L else None
    elif scan_id == 1: # cols, forward
        lines = [[(r, c) for r in range(L)] for c in range(L)]
        next_start = lambda c: (0, c + 1) if c + 1 < L else None
    elif scan_id == 2: # rows, backward
        lines = [[(r, c) for c in range(L - 1, -1, -1)] for r in range(L - 1, -1, -1)]
        next_start = lambda r: (r - 1, L - 1) if r - 1 >= 0 else None
    else:              # cols, backward
        lines = [[(r, c) for r in range(L - 1, -1, -1)] for c in range(L - 1, -1, -1)]
        next_start = lambda c: (L - 1, c - 1) if c - 1 >= 0 else None

    for li, line in enumerate(lines):
        # solid arrow spanning the whole line
        ann_arrow(ax, cell_xy(*line[0]), cell_xy(*line[-1]),
                  color=color, lw=2.0, rad=0.0)
        # dashed return to start of next line
        ns = next_start(li)
        if ns is not None:
            ann_arrow(ax, cell_xy(*line[-1]), cell_xy(*ns),
                      color="#555", lw=0.9, rad=-0.4, dashed=True, ms=10)

    dot(ax, cell_xy(*seq[0]),  color="#27ae60")   # start
    dot(ax, cell_xy(*seq[-1]), color=color)        # end

    ax.plot([], [], color=color, lw=2, label="scan direction")
    ax.plot([], [], color="#555", lw=1, ls="--", label="return jump")
    ax.scatter([], [], s=55, color="#27ae60", label="start")
    ax.scatter([], [], s=55, color=color,     label="end")


def make_ss2d_figures(rp, out_dir):
    labels = ["A_row_fwd", "B_col_fwd", "C_row_bwd", "D_col_bwd"]
    for i in range(4):
        fig, ax = plt.subplots(figsize=FIG_SZ)
        _ss2d_panel(ax, rp, i)
        # reserve bottom space for legend, then place it below
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        handles, lbls = ax.get_legend_handles_labels()
        fig.legend(handles, lbls,
                   loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, 0.01),
                   framealpha=0.9, edgecolor="none",
                   fontsize=FONT - 1)
        save(fig, out_dir, f"fig_ss2d_scan_{labels[i]}")
        print(f"  saved fig_ss2d_scan_{labels[i]}")


# ═══════════════════════════════════════════════════════════════════════════════
#  UpperTriDiag — 2 panels
# ═══════════════════════════════════════════════════════════════════════════════

def make_upper_tri_pass1(ax, rp):
    """Pass 1: show the RP grid with upper-triangle anti-diagonals coloured by lag.
    Lower triangle is masked in red. This is the *extraction* step.
    Returns legend_handles for placing outside the axes."""
    draw_grid_bg(ax, rp, alpha=0.12)
    ax.set_title("Pass 1 — Extract anti-diagonal\nlag tokens from upper triangle", pad=8)

    n_lags = L - 1
    lag_colors = [LAG_CMAP(k / n_lags) for k in range(n_lags)]

    legend_handles = []

    for k in range(1, L):       # lag k = 1 … L-1
        color = lag_colors[k - 1]
        rows = list(range(L - k))
        cols = [r + k for r in rows]

        # colour cells
        for r, c in zip(rows, cols):
            rect = mpatches.FancyBboxPatch(
                (c + 0.07, -(r + 0.93)), 0.86, 0.86,
                boxstyle="round,pad=0.03",
                linewidth=1.4, edgecolor=color,
                facecolor=mcolors.to_rgba(color, 0.42), zorder=2)
            ax.add_patch(rect)

        # step-by-step arrows along the anti-diagonal
        if len(rows) >= 2:
            for step in range(len(rows) - 1):
                ann_arrow(ax,
                          cell_xy(rows[step],     cols[step]),
                          cell_xy(rows[step + 1], cols[step + 1]),
                          color=color, lw=2.0, rad=0.0, ms=12)
            dot(ax, cell_xy(rows[0], cols[0]), color=color, size=50)

        # dashed return to next lag start
        if k < L - 1:
            ann_arrow(ax,
                      cell_xy(rows[-1], cols[-1]),
                      cell_xy(0, k + 1),
                      color=color, lw=0.9, rad=-0.38, dashed=True, ms=9)

        legend_handles.append(
            mpatches.Patch(facecolor=mcolors.to_rgba(color, 0.7),
                           edgecolor=color, label=f"lag {k}"))

    # red lower triangle overlay
    lower = np.zeros((L, L))
    for r in range(L):
        for c in range(r):
            lower[r, c] = 1.0
    ax.imshow(lower,
              cmap=mcolors.LinearSegmentedColormap.from_list(
                  "rt", [(1,1,1,0), (1, 0, 0, 0.22)]),
              extent=[0, L, -L, 0], aspect="equal", origin="upper", zorder=1)

    # diagonal marker
    ax.plot([c + 0.5 for c in range(L)],
            [-(r + 0.5) for r in range(L)],
            color="grey", lw=1.5, ls=":", zorder=3)

    legend_handles.append(
        mpatches.Patch(facecolor=(1, 0, 0, 0.22), edgecolor="none",
                       label="unused (lower tri)"))
    return legend_handles


def make_upper_tri_pass2(ax, rp):
    """Pass 2: show the flattened lag-token sequence fed into Mamba left→right.
    Each token is a rectangle coloured by lag; an arrow sweeps through them."""
    ax.set_aspect("auto")
    ax.set_title("Pass 2 — Mamba scans the\nlag-token sequence (lag 1 → L−1)", pad=8)

    n_lags = L - 1
    lag_colors = [LAG_CMAP(k / n_lags) for k in range(n_lags)]

    # Draw one rectangle per lag token in a horizontal strip
    token_w = 0.9
    token_h = 2.2
    gap = 0.18
    y0 = 0.0   # bottom of strip

    total_w = n_lags * (token_w + gap) - gap

    for k in range(1, L):
        color = lag_colors[k - 1]
        x0 = (k - 1) * (token_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x0, y0), token_w, token_h,
            boxstyle="round,pad=0.06",
            linewidth=1.4, edgecolor=color,
            facecolor=mcolors.to_rgba(color, 0.50),
            zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + token_w / 2, y0 + token_h / 2,
                f"lag\n{k}",
                ha="center", va="center",
                fontsize=FONT - 3, color="black", zorder=3)
        # small number of elements label
        n_elems = L - k
        ax.text(x0 + token_w / 2, y0 - 0.38,
                f"n={n_elems}",
                ha="center", va="top",
                fontsize=FONT - 5, color="#555", zorder=3)

    # Big arrow sweeping left → right across all tokens
    ax.annotate(
        "", xy=(total_w + 0.15, y0 + token_h / 2),
        xytext=(-0.3, y0 + token_h / 2),
        arrowprops=dict(arrowstyle="-|>", color="#c0392b",
                        mutation_scale=18, lw=2.4),
        zorder=4)

    # Label: Mamba SSM
    ax.text(total_w / 2, y0 + token_h + 0.55,
            "Mamba SSM →",
            ha="center", va="bottom",
            fontsize=FONT, color="#c0392b", weight="bold")

    # Pool arrow below
    ax.annotate(
        "", xy=(total_w / 2, y0 - 1.5),
        xytext=(total_w / 2, y0 - 0.6),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                        mutation_scale=16, lw=2.0),
        zorder=4)
    ax.text(total_w / 2, y0 - 1.7,
            "mean pool → embedding",
            ha="center", va="top",
            fontsize=FONT - 2, color="#2c3e50")

    ax.set_xlim(-0.6, total_w + 0.6)
    ax.set_ylim(y0 - 2.4, y0 + token_h + 1.2)
    ax.axis("off")


def make_upper_tri_figures(rp, out_dir):
    # Panel 1: extraction — legend + caption outside
    fig, ax = plt.subplots(figsize=FIG_SZ)
    leg_handles = make_upper_tri_pass1(ax, rp)
    # reserve bottom 28% for legend + caption, then place both
    fig.tight_layout(rect=[0, 0.28, 1, 1])
    fig.legend(handles=leg_handles,
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.12),
               framealpha=0.9, edgecolor="none",
               fontsize=FONT - 2)
    fig.text(0.5, 0.04,
             "← anti-diagonals extracted in lag order (lag 1 → lag L−1)",
             ha="center", va="bottom",
             fontsize=FONT - 2, color="#444", style="italic")
    save(fig, out_dir, "fig_upper_tri_pass1_extraction")
    print("  saved fig_upper_tri_pass1_extraction")

    # Panel 2: Mamba scan
    fig, ax = plt.subplots(figsize=(FIG_SZ[0] + 1, FIG_SZ[1] * 0.55))
    make_upper_tri_pass2(ax, rp)
    fig.tight_layout()
    save(fig, out_dir, "fig_upper_tri_pass2_mamba")
    print("  saved fig_upper_tri_pass2_mamba")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results" / "triangu_paper" / "scan_steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    rp = make_rp(L)

    print("── SS2D 4-scan figures ──")
    make_ss2d_figures(rp, out_dir)

    print("── UpperTriDiag 2-pass figures ──")
    make_upper_tri_figures(rp, out_dir)

    print(f"\nAll done → {out_dir}")


if __name__ == "__main__":
    main()
