#!/usr/bin/env python3
"""Generate ICML dataset time-series example plots and append to markdown.

Creates one figure per dataset with 3 example univariate series (feature columns),
then writes a markdown gallery and injects it into results/SSL_METHOD_COMPARISON.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class DatasetSpec:
    name: str
    path: Path


def pick_three_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 3:
        raise ValueError(f"Need >=3 numeric columns, found {len(numeric_cols)}")
    idx = [0, len(numeric_cols) // 2, len(numeric_cols) - 1]
    cols = [numeric_cols[i] for i in idx]
    # Avoid duplicates when len is exactly 3 or edge cases
    seen = []
    for c in cols:
        if c not in seen:
            seen.append(c)
    if len(seen) < 3:
        for c in numeric_cols:
            if c not in seen:
                seen.append(c)
            if len(seen) == 3:
                break
    return seen[:3]


def plot_dataset_examples(spec: DatasetSpec, out_dir: Path, max_points: int) -> Path:
    df = pd.read_csv(spec.path)
    cols = pick_three_columns(df)

    n = min(max_points, len(df))
    x = range(n)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"{spec.name}: 3 example time series (first {n} points)", fontsize=12)

    for i, col in enumerate(cols):
        y = df[col].iloc[:n]
        axes[i].plot(x, y, linewidth=1.0)
        axes[i].set_ylabel(col)
        axes[i].grid(alpha=0.25)

    axes[-1].set_xlabel("time index")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_dir / f"{spec.name}_3_examples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_gallery_markdown(rel_plot_paths: List[Path]) -> str:
    lines: List[str] = []
    lines.append("## ICML Datasets — 3 Time-Series Examples per Dataset")
    lines.append("")
    lines.append("Each figure shows 3 feature columns from the same dataset (first window of points).")
    lines.append("")
    for p in rel_plot_paths:
        dataset = p.stem.replace("_3_examples", "")
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append(f"![{dataset} examples]({p.as_posix()})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def inject_section(target_md: Path, section_md: str) -> None:
    start = "<!-- ICML_SERIES_EXAMPLES:START -->"
    end = "<!-- ICML_SERIES_EXAMPLES:END -->"

    block = f"{start}\n\n{section_md}\n{end}\n"

    text = target_md.read_text(encoding="utf-8") if target_md.exists() else ""
    if start in text and end in text:
        pre = text.split(start)[0]
        post = text.split(end, 1)[1]
        new_text = pre + block + post.lstrip("\n")
    else:
        sep = "\n\n" if text and not text.endswith("\n\n") else ""
        new_text = text + sep + block

    target_md.write_text(new_text, encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    icml = root / "ICML_datasets"
    out_dir = root / "results" / "icml_series_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        DatasetSpec("ETTh1", icml / "ETT-small" / "ETTh1.csv"),
        DatasetSpec("ETTh2", icml / "ETT-small" / "ETTh2.csv"),
        DatasetSpec("ETTm1", icml / "ETT-small" / "ETTm1.csv"),
        DatasetSpec("ETTm2", icml / "ETT-small" / "ETTm2.csv"),
        DatasetSpec("electricity", icml / "electricity" / "electricity.csv"),
        DatasetSpec("exchange_rate", icml / "exchange_rate" / "exchange_rate.csv"),
        DatasetSpec("traffic", icml / "traffic" / "traffic.csv"),
        DatasetSpec("weather", icml / "weather" / "weather.csv"),
    ]

    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"Missing dataset file: {spec.path}")

    generated: List[Path] = []
    for spec in specs:
        generated.append(plot_dataset_examples(spec, out_dir, max_points=1000))

    rel_paths = [p.relative_to(root / "results") for p in generated]
    section_md = build_gallery_markdown(rel_paths)

    gallery_md = out_dir / "ICML_SERIES_EXAMPLES.md"
    gallery_md.write_text(section_md, encoding="utf-8")

    target_md = root / "results" / "SSL_METHOD_COMPARISON.md"
    inject_section(target_md, section_md)

    print(f"Generated {len(generated)} plot files in: {out_dir}")
    print(f"Updated markdown section in: {target_md}")


if __name__ == "__main__":
    main()
