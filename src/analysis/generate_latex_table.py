"""
Generate LaTeX comparison tables for SSL methods — micro and nano tiers.

Two tables are produced:
  1. Micro tier  (icml_*_micro  results, avg MSE+MAE per dataset)
  2. Nano tier   (probe_*_nano  results, avg MSE+MAE per dataset)

Data merging logic:
  - Base CSV: probe_results_full.csv or probe_lotsa_results.csv in the main folder
  - Solar supplement: *_solar/ folder — merged in for solar_AL dataset
  - PSN supplement:   *_psn/   folder — overrides exchange_rate rows when present
    (currently disabled: old PSN had Y/X scale mismatch; fix implemented, new jobs needed)

Colour rules (applied per row / per metric independently):
  - Best value   → red bold       \\textcolor{red}{\\textbf{x}}
  - Second best  → blue underline \\textcolor{blue}{\\underline{x}}
  - Missing data → --

Usage:
    python src/analysis/generate_latex_table.py
    python src/analysis/generate_latex_table.py --tier micro
    python src/analysis/generate_latex_table.py --tier nano
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"

# Base result folder per method.
MICRO_METHODS: Dict[str, str] = {
    "SimCLR":  "icml_simclr_micro",
    "CLIP":    "icml_clip_micro",
    "BYOL":    "icml_byol_micro",
    "VL-JEPA": "icml_vl_jepa_micro",
    "GRAM":    "icml_gram_micro",
}

NANO_METHODS: Dict[str, str] = {
    "SimCLR-Bi": "probe_simclr_bimodal_nano",
    "CLIP":      "probe_clip_nano",
    "BYOL-Bi":   "probe_byol_bimodal_nano",
    "VL-JEPA":   "probe_vl_jepa_nano",
    "GRAM":      "probe_gram_nano",
}

# Supplementary result folders: dataset_key → suffix appended to base folder name.
# e.g. "icml_simclr_micro" + "_solar" → "icml_simclr_micro_solar"
SUPPLEMENTS: Dict[str, str] = {
    "solar_AL":      "_solar",   # solar_AL probe saved under *_solar/
    "exchange_rate": "_psn",     # symmetric PSN (X and Y normalised together) — much better than base
}

DATASET_ORDER: List[str] = [
    "ETTm1", "ETTm2", "ETTh1", "ETTh2",
    "weather", "traffic", "electricity", "exchange_rate", "solar_AL",
]

DATASET_DISPLAY: Dict[str, str] = {
    "ETTm1":         "ETTm1",
    "ETTm2":         "ETTm2",
    "ETTh1":         "ETTh1",
    "ETTh2":         "ETTh2",
    "weather":       "Weather",
    "traffic":       "Traffic",
    "electricity":   "Electricity",
    "exchange_rate": "Exch. Rate",
    "solar_AL":      "Solar",
}

HORIZONS: List[int] = [96, 192, 336, 720]

# CSV filename candidates, tried in order (newest / most complete first)
CSV_CANDIDATES = ["probe_results_full.csv", "probe_lotsa_results.csv"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_csv(folder: Path) -> Optional[Path]:
    for name in CSV_CANDIDATES:
        p = folder / name
        if p.exists():
            return p
    return None


def _read_csv(path: Path) -> Dict[Tuple[str, int], Tuple[float, float]]:
    rows: Dict[Tuple[str, int], Tuple[float, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ds  = row["dataset"].strip()
            h   = int(row["horizon"])
            mse = float(row["mse"])
            mae = float(row["mae"])
            rows[(ds, h)] = (mse, mae)
    return rows


def load_tier(
    methods: Dict[str, str],
) -> Dict[str, Dict[Tuple[str, int], Tuple[float, float]]]:
    """Load and merge all CSVs for a tier.

    For each method:
      1. Load base CSV from the main folder.
      2. For each dataset in SUPPLEMENTS, look for a *_<suffix>/ folder and
         merge those rows in (overriding base rows for that dataset).
    """
    data: Dict[str, Dict[Tuple[str, int], Tuple[float, float]]] = {}

    for name, folder in methods.items():
        rows: Dict[Tuple[str, int], Tuple[float, float]] = {}

        # ── 1. base CSV ──
        base_csv = _find_csv(RESULTS_ROOT / folder)
        if base_csv is None:
            print(f"  WARNING: no base CSV for {name} in results/{folder}/")
        else:
            rows.update(_read_csv(base_csv))
            print(f"  Loaded {name}: {len(rows)} rows from {base_csv.parent.name}/{base_csv.name}")

        # ── 2. supplements ──
        for ds_key, suffix in SUPPLEMENTS.items():
            supp_folder = RESULTS_ROOT / (folder + suffix)
            supp_csv = _find_csv(supp_folder)
            if supp_csv is None:
                # Also try alternate naming for nano (e.g. simclr_bi_nano_solar)
                alt_name = folder.replace("probe_", "").replace("_nano", "") + "_nano" + suffix
                supp_csv = _find_csv(RESULTS_ROOT / alt_name)
            if supp_csv is None:
                print(f"  -- {name}: no supplement for '{ds_key}' (looked in {folder+suffix}/)")
                continue
            supp_rows = _read_csv(supp_csv)
            # Only take rows matching this dataset key
            merged = 0
            for (ds, h), vals in supp_rows.items():
                if ds == ds_key or ds.startswith(ds_key):
                    rows[(ds, h)] = vals
                    merged += 1
            if merged:
                print(f"  Merged {name} supplement '{ds_key}': {merged} rows from {supp_csv.parent.name}/")

        data[name] = rows

    return data


def avg_over_horizons(
    data: Dict[str, Dict[Tuple[str, int], Tuple[float, float]]],
    methods: List[str],
) -> Dict[str, Dict[str, Optional[Tuple[float, float]]]]:
    """Return {method: {dataset: (avg_mse, avg_mae) or None}}."""
    out: Dict[str, Dict[str, Optional[Tuple[float, float]]]] = {}
    for m in methods:
        out[m] = {}
        for ds in DATASET_ORDER:
            vals = [data[m][(ds, h)] for h in HORIZONS if (ds, h) in data[m]]
            if vals:
                avg_mse = sum(v[0] for v in vals) / len(vals)
                avg_mae = sum(v[1] for v in vals) / len(vals)
                out[m][ds] = (avg_mse, avg_mae)
            else:
                out[m][ds] = None
    return out

# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _cell(v: float, rank: int) -> str:
    s = _fmt(v)
    if rank == 0:
        return r"\textcolor{red}{\textbf{" + s + r"}}"
    if rank == 1:
        return r"\textcolor{blue}{\underline{" + s + r"}}"
    return s


def _rank_cells(method_vals: Dict[str, Optional[float]]) -> Dict[str, int]:
    """Assign rank 0=best, 1=second, 2+=rest. Missing → sentinel 999."""
    present = {m: v for m, v in method_vals.items() if v is not None}
    sorted_m = sorted(present, key=lambda m: present[m])
    ranks = {m: i for i, m in enumerate(sorted_m)}
    for m in method_vals:
        if m not in ranks:
            ranks[m] = 999
    return ranks

# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(
    avgs: Dict[str, Dict[str, Optional[Tuple[float, float]]]],
    methods: List[str],
    caption: str,
    label: str,
) -> str:
    n = len(methods)
    col_spec = "l" + "cc" * n

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # header row 1: method names
    method_headers = " & ".join(
        r"\multicolumn{2}{c}{\textbf{" + m + r"}}" for m in methods
    )
    lines.append(r"\textbf{Dataset} & " + method_headers + r" \\")

    # cmidrules
    cmidrules = []
    for i in range(n):
        lo = 2 + i * 2
        hi = lo + 1
        cmidrules.append(r"\cmidrule(lr){" + str(lo) + r"-" + str(hi) + r"}")
    lines.append("".join(cmidrules))

    # header row 2: MSE / MAE
    lines.append(r" & " + " & ".join(r"MSE & MAE" for _ in methods) + r" \\")
    lines.append(r"\midrule")

    # data rows
    for ds in DATASET_ORDER:
        ds_label = DATASET_DISPLAY[ds]

        mse_vals = {m: avgs[m][ds][0] if avgs[m][ds] is not None else None for m in methods}
        mae_vals = {m: avgs[m][ds][1] if avgs[m][ds] is not None else None for m in methods}

        mse_ranks = _rank_cells(mse_vals)
        mae_ranks = _rank_cells(mae_vals)

        cells = []
        for m in methods:
            if mse_vals[m] is None:
                cells.append(r"-- & --")
            else:
                cells.append(
                    _cell(mse_vals[m], mse_ranks[m]) + " & " +
                    _cell(mae_vals[m], mae_ranks[m])
                )

        lines.append(ds_label + " & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_summary(avgs: Dict, methods: Dict[str, str]) -> None:
    col_w = 14
    print(f"\n{'─'*60}")
    print(f"{'Dataset':<{col_w}}" + "".join(f"  {m:<14}" for m in methods))
    print(f"{'─'*60}")
    for ds in DATASET_ORDER:
        row = f"{DATASET_DISPLAY[ds]:<{col_w}}"
        for m in methods:
            v = avgs[m][ds]
            row += f"  {v[0]:.3f}/{v[1]:.3f}" if v else "  --/--        "
        print(row)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["micro", "nano", "both"], default="both")
    args = parser.parse_args()

    output_parts = []
    output_parts.append("% Required packages: booktabs, multirow, xcolor\n")
    output_parts.append("% Auto-generated by src/analysis/generate_latex_table.py\n")
    output_parts.append("% Solar rows: merged from *_solar/ supplements\n")
    output_parts.append("% Exchange Rate rows: merged from *_psn/ supplements (per-series norm)\n\n")

    tiers = []
    if args.tier in ("micro", "both"):
        tiers.append(("micro", MICRO_METHODS, "tab:micro_ssl"))
    if args.tier in ("nano", "both"):
        tiers.append(("nano",  NANO_METHODS,  "tab:nano_ssl"))

    for tier_name, methods_cfg, label in tiers:
        print(f"\n── {tier_name.capitalize()} tier ──")
        data = load_tier(methods_cfg)
        methods = list(methods_cfg.keys())
        avgs = avg_over_horizons(data, methods)

        table = build_table(
            avgs,
            methods,
            caption=(
                rf"Linear probe performance — \textbf{{{tier_name}}} tier "
                r"(avg MSE/MAE over H$\in$\{96,192,336,720\}). "
                r"Exchange Rate uses per-series normalisation. "
                r"\textcolor{red}{\textbf{Red}}: best per dataset, "
                r"\textcolor{blue}{\underline{Blue}}: second-best."
            ),
            label=label,
        )
        output_parts.append(table)
        output_parts.append("\n\n")
        print_summary(avgs, methods_cfg)

    out_path = RESULTS_ROOT / "ssl_probe_tables.tex"
    with open(out_path, "w") as f:
        f.write("".join(output_parts))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
