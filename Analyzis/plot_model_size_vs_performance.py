"""Bubble chart: model size vs ETTm performance.

Generates a figure similar to the provided reference:
- X axis: model parameters (millions), log-scaled
- Y axis: MSE (lower is better)
- Color: unique per model/variant
- Bubble area: scaled from parameter count

Usage:
  python analysis/plot_model_size_vs_performance.py --metric avg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ModelPoint:
    model: str
    variant: str
    params_m: float
    ettm1_mse: float
    ettm2_mse: float

    @property
    def avg_ettm_mse(self) -> float:
        return 0.5 * (self.ettm1_mse + self.ettm2_mse)


def _default_points() -> List[ModelPoint]:
    # Values provided by the user.
    return [
        ModelPoint("CM-Mamba", "CM-Mamba-Tiny", 1.2, 0.192, 0.199),
        ModelPoint("S-Mamba", "S-Mamba", 0.5, 0.375, 0.276),
        ModelPoint("LightGTS", "LightGTS-Tiny", 1.3, 0.345, 0.249),
        ModelPoint("Chronos", "Chronos-Tiny", 9.0, 0.551, 0.293),
        # Moirai values are marked with * in the table; we keep the numeric values for plotting.
        ModelPoint("Moirai 2.0", "Moirai-Small", 11.0, 0.390, 0.290),
        ModelPoint("Time-MOE", "Time-MOE Base", 50.0, 0.345, 0.271),
        ModelPoint("Timer", "Timer-Base", 84.0, 0.768, 0.315),
        ModelPoint("TimesFM", "TimesFM", 200.0, 0.435, 0.347),
    ]


def _compute_bubble_sizes(
    params_m: np.ndarray,
    *,
    min_s: float,
    max_s: float,
    domain_min: float | None = None,
    domain_max: float | None = None,
) -> np.ndarray:
    """Compute scatter marker areas from parameter counts.

    Uses sqrt-scaling to avoid the largest model dominating the plot while
    preserving monotonicity.
    """

    params_m = np.asarray(params_m, dtype=float)
    params_m = np.maximum(params_m, 1e-9)

    if domain_min is None:
        domain_min = float(np.min(params_m))
    if domain_max is None:
        domain_max = float(np.max(params_m))
    domain_min = max(float(domain_min), 1e-9)
    domain_max = max(float(domain_max), domain_min)

    scaled = np.sqrt(params_m)
    scaled_min = float(np.sqrt(domain_min))
    scaled_max = float(np.sqrt(domain_max))
    denom = float(scaled_max - scaled_min)
    if denom <= 0:
        return np.full_like(scaled, 0.5 * (min_s + max_s))

    alpha = (scaled - scaled_min) / (denom + 1e-12)
    return min_s + alpha * (max_s - min_s)


def _pick_metric(points: Iterable[ModelPoint], metric: str) -> Tuple[np.ndarray, str]:
    metric = metric.lower().strip()
    if metric == "avg":
        return np.array([p.avg_ettm_mse for p in points], dtype=float), "Average MSE"
    if metric == "ettm1":
        return np.array([p.ettm1_mse for p in points], dtype=float), "ETTm1 MSE"
    if metric == "ettm2":
        return np.array([p.ettm2_mse for p in points], dtype=float), "ETTm2 MSE"
    raise ValueError("metric must be one of: avg, ettm1, ettm2")


def plot(
    points: List[ModelPoint],
    *,
    metric: str,
    title: str,
    output_path: Path,
    dpi: int,
    annotate: bool,
    show_legend: bool,
    figsize: Tuple[float, float],
) -> Path:
    # ICML-style academic publication formatting
    plt.style.use("classic")
    plt.rcParams["figure.dpi"] = dpi
    # Clean white backgrounds for publication
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "none"
    # Academic typography - serif font like Times New Roman
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["legend.title_fontsize"] = 14
    # Publication-quality line widths
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.5
    # Better rendering for vector formats
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    x = np.array([p.params_m for p in points], dtype=float)
    y, y_label = _pick_metric(points, metric)

    sizes = _compute_bubble_sizes(x, min_s=150.0, max_s=1400.0, domain_min=float(x.min()), domain_max=float(x.max()))

    # Professional color palette for academic publications
    # Using distinct, colorblind-friendly colors
    academic_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    colors = [academic_colors[i % len(academic_colors)] for i in range(len(points))]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # Add a little axis padding up-front (then we finalize limits after).
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    # log-scale padding: multiplicative expansion
    x_pad_left = 1.35
    x_pad_right = 1.35
    ax.set_xlim(max(1e-6, x_min / x_pad_left), x_max * x_pad_right)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_pad = max(0.01, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    x_mid = float(np.sqrt(x_min * x_max))  # geometric mid for log-scale
    y_mid = 0.5 * (y_min + y_max)

    for idx, p in enumerate(points):
        ax.scatter(
            [p.params_m],
            [y[idx]],
            s=float(sizes[idx]),
            color=colors[idx],
            edgecolors="#2d2d2d",
            linewidths=0.8,
            alpha=0.75,
            label=p.model,
            zorder=3,
        )

        if annotate:
            # Place labels close to bubbles but avoid clipping near boundaries.
            place_left = float(p.params_m) >= x_mid
            place_below = float(y[idx]) >= y_mid
            dx = -10 if place_left else 10
            dy = -10 if place_below else 10
            ha = "right" if place_left else "left"
            va = "top" if place_below else "bottom"
            ax.annotate(
                p.model,
                (p.params_m, y[idx]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=24,
                fontweight="normal",
                color="#1a1a1a",
                zorder=4,
                annotation_clip=False,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters (Millions)", fontweight="normal", fontsize=24)
    ax.set_ylabel(y_label, fontweight="normal", fontsize=24)
    if title:
        ax.set_title(title, fontweight="normal", fontsize=24)

    # Add subtle grid for academic readability
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    
    # Professional border around the plot area
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d2d2d')
        spine.set_linewidth(1.0)
        spine.set_visible(True)

    if show_legend:
        # Legend for colors (models)
        legend_models = ax.legend(
            title="Models",
            loc="upper left",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            ncol=1,
            framealpha=0.95,
            edgecolor="#2d2d2d",
            fancybox=False,
        )

        # Bubble size legend (parameter counts)
        size_refs = np.array([1.0, 10.0, 100.0])
        size_handles = [
            plt.scatter(
                [],
                [],
                s=float(
                    _compute_bubble_sizes(
                        np.array([v]),
                        min_s=150.0,
                        max_s=1400.0,
                        domain_min=float(x.min()),
                        domain_max=float(x.max()),
                    )[0]
                ),
                color="#888888",
                edgecolors="#2d2d2d",
                linewidths=0.8,
                alpha=0.7,
            )
            for v in size_refs
        ]
        size_labels = [f"{v:g}M" for v in size_refs]
        legend_sizes = ax.legend(
            size_handles,
            size_labels,
            title="Params",
            loc="lower right",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            framealpha=0.95,
            edgecolor="#2d2d2d",
            fancybox=False,
        )

        ax.add_artist(legend_models)

    fig.tight_layout(pad=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", format='png')
    
    # Save as PDF for publication (vector format)
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches="tight", facecolor="white")
    print(f"Saved PDF: {pdf_path}")
    
    # Save as EPS for publication (vector format)
    eps_path = output_path.with_suffix('.eps')
    fig.savefig(eps_path, format='eps', bbox_inches="tight", facecolor="white")
    print(f"Saved EPS: {eps_path}")
    
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metric",
        default="avg",
        choices=["avg", "ettm1", "ettm2"],
        help="Which metric to plot on y-axis.",
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "model_size_vs_performance.png"),
        help="Output image path.",
    )
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    p.add_argument(
        "--title",
        default="",
        help="Plot title.",
    )
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Disable plot title.",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(12.0, 6.2),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        default=True,
        help="Add point labels next to each bubble.",
    )
    p.add_argument(
        "--legend",
        action="store_true",
        help="Show legends (models and bubble size).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    points = _default_points()

    title = "" if bool(args.no_title) else str(args.title)

    plot(
        points,
        metric=args.metric,
        title=title,
        output_path=output_path,
        dpi=int(args.dpi),
        annotate=bool(args.annotate),
        show_legend=bool(args.legend),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()