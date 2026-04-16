#!/usr/bin/env python3
"""
Generate LaTeX table comparing CLIP unimodal temporal, unimodal visual,
and CLIP multimodal contrastive (micro ICML) linear probe results.

Usage:
    python3 generate_unimodal_comparison_table.py [--out results/table_unimodal_comparison.tex]

CSV sources (resolved relative to project root):
    Temporal : results/clip_temporal/probe_lotsa_results.csv
    Visual   : results/clip_visual/probe_results_full.csv
    Multimodal: results/icml_clip_micro/probe_results_full.csv

Solar/PEMS rows are included if present in all three CSVs; otherwise skipped.
"""

import argparse
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULT_CSVS = {
    "T":  PROJECT_ROOT / "results/clip_temporal/probe_lotsa_results.csv",
    "V":  PROJECT_ROOT / "results/clip_visual/probe_results_full.csv",
    "CM": PROJECT_ROOT / "results/icml_clip_micro/probe_results_full.csv",
}

# Display order and label for each dataset key (as it appears in the CSVs)
DATASET_ORDER = [
    ("ETTm1",         "ETTm1"),
    ("ETTm2",         "ETTm2"),
    ("ETTh1",         "ETTh1"),
    ("ETTh2",         "ETTh2"),
    ("traffic",       "Traffic"),
    ("weather",       "Weather"),
    ("exchange_rate", "Exchange"),
    ("solar_AL",      "Solar"),
    ("electricity",   "Electricity"),
    ("PEMS03",        "PEMS03"),
    ("PEMS04",        "PEMS04"),
    ("PEMS07",        "PEMS07"),
    ("PEMS08",        "PEMS08"),
]

HORIZONS = [96, 192, 336, 720]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> dict:
    """Return dict: {dataset_key: {horizon: (mse, mae)}}"""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            h  = int(row["horizon"])
            mse = float(row["mse"])
            mae = float(row["mae"])
            data.setdefault(ds, {})[h] = (mse, mae)
    return data


def rank3(a, b, c):
    """Return (rank_a, rank_b, rank_c): 1=best(lowest MSE/MAE), 2=second, 3=worst."""
    vals = [(a, 0), (b, 1), (c, 2)]
    vals_sorted = sorted(vals, key=lambda x: x[0])
    ranks = [0, 0, 0]
    ranks[vals_sorted[0][1]] = 1
    ranks[vals_sorted[1][1]] = 2
    ranks[vals_sorted[2][1]] = 3
    return ranks


def fmt(val: float, rank: int) -> str:
    s = f"{val:.4f}"
    if rank == 1:
        return rf"\textcolor{{red}}{{\textbf{{{s}}}}}"
    elif rank == 2:
        return rf"\textcolor{{blue}}{{\underline{{{s}}}}}"
    return s


def mean(lst):
    return sum(lst) / len(lst)


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(t_data: dict, v_data: dict, cm_data: dict) -> str:
    lines = []

    # Preamble
    lines += [
        r"\begin{table}[t!]",
        r"\centering",
        r"\caption{Linear-probe evaluation comparing CLIP unimodal temporal, unimodal visual,"
        r" and contrastive multimodal (CLIP micro) encoders."
        r" \textcolor{red}{\textbf{Red}}: best, \textcolor{blue}{\underline{Blue}}: 2nd best.}",
        r"\small",
        r"\begin{adjustbox}{max width=\columnwidth}",
        r"\label{tab:contrastive_multimodal_comparison}",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"\textbf{Models} & \textbf{Metric}"
        r" & \multicolumn{2}{c}{\textbf{Temporal}}"
        r" & \multicolumn{2}{c}{\textbf{Visual}}"
        r" & \multicolumn{2}{c}{\textbf{Contrastive Multimodal}} \\",
        r"\cmidrule(lr){3-4}",
        r"\cmidrule(lr){5-6}",
        r"\cmidrule(lr){7-8}",
        r" & & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} & \textbf{MSE} & \textbf{MAE} \\",
        r"\midrule",
    ]

    best_counts  = {"T": 0, "V": 0, "CM": 0}
    second_counts = {"T": 0, "V": 0, "CM": 0}

    for ds_key, ds_label in DATASET_ORDER:
        # Skip dataset if missing from any CSV
        if ds_key not in t_data or ds_key not in v_data or ds_key not in cm_data:
            continue
        # Skip if any horizon is missing
        missing = False
        for h in HORIZONS:
            if h not in t_data[ds_key] or h not in v_data[ds_key] or h not in cm_data[ds_key]:
                missing = True
                break
        if missing:
            continue

        rows = []
        for h in HORIZONS:
            t_mse, t_mae = t_data[ds_key][h]
            v_mse, v_mae = v_data[ds_key][h]
            c_mse, c_mae = cm_data[ds_key][h]

            r_mse = rank3(t_mse, v_mse, c_mse)
            r_mae = rank3(t_mae, v_mae, c_mae)

            # Count best/second
            keys = ["T", "V", "CM"]
            for i, k in enumerate(keys):
                if r_mse[i] == 1: best_counts[k] += 1
                elif r_mse[i] == 2: second_counts[k] += 1
                if r_mae[i] == 1: best_counts[k] += 1
                elif r_mae[i] == 2: second_counts[k] += 1

            rows.append((h, t_mse, t_mae, v_mse, v_mae, c_mse, c_mae, r_mse, r_mae))

        # Mean row
        t_mse_m = mean([t_data[ds_key][h][0] for h in HORIZONS])
        t_mae_m = mean([t_data[ds_key][h][1] for h in HORIZONS])
        v_mse_m = mean([v_data[ds_key][h][0] for h in HORIZONS])
        v_mae_m = mean([v_data[ds_key][h][1] for h in HORIZONS])
        c_mse_m = mean([cm_data[ds_key][h][0] for h in HORIZONS])
        c_mae_m = mean([cm_data[ds_key][h][1] for h in HORIZONS])
        r_mse_m = rank3(t_mse_m, v_mse_m, c_mse_m)
        r_mae_m = rank3(t_mae_m, v_mae_m, c_mae_m)

        # Emit rows
        nrows = len(rows)
        for idx, (h, t_mse, t_mae, v_mse, v_mae, c_mse, c_mae, r_mse, r_mae) in enumerate(rows):
            prefix = (
                rf" \multirow{{{nrows + 1}}}{{*}}{{\textbf{{{ds_label}}}}} & {h}"
                if idx == 0 else f"  & {h}"
            )
            lines.append(
                f"{prefix} & {fmt(t_mse, r_mse[0])} & {fmt(t_mae, r_mae[0])}"
                f" & {fmt(v_mse, r_mse[1])} & {fmt(v_mae, r_mae[1])}"
                f" & {fmt(c_mse, r_mse[2])} & {fmt(c_mae, r_mae[2])}\\\\"
            )

        lines.append(r"  \cmidrule{2-8}")
        lines.append(
            f"  & \\textbf{{Mean}}"
            f" & {fmt(t_mse_m, r_mse_m[0])} & {fmt(t_mae_m, r_mae_m[0])}"
            f" & {fmt(v_mse_m, r_mse_m[1])} & {fmt(v_mae_m, r_mae_m[1])}"
            f" & {fmt(c_mse_m, r_mse_m[2])} & {fmt(c_mae_m, r_mae_m[2])}\\\\"
        )
        lines.append(r"\midrule")

    # Best / second-best count footer
    lines += [
        r"\textbf{Best count} & "
        rf"& \multicolumn{{2}}{{c}}{{\textcolor{{red}}{{\textbf{{{best_counts['T']}}}}}}} "
        rf"& \multicolumn{{2}}{{c}}{{\textcolor{{red}}{{\textbf{{{best_counts['V']}}}}}}} "
        rf"& \multicolumn{{2}}{{c}}{{\textcolor{{red}}{{\textbf{{{best_counts['CM']}}}}}}} \\",
        r"\textbf{Second-best count} & "
        rf"& \multicolumn{{2}}{{c}}{{\textcolor{{blue}}{{\underline{{{second_counts['T']}}}}}}} "
        rf"& \multicolumn{{2}}{{c}}{{\textcolor{{blue}}{{\underline{{{second_counts['V']}}}}}}} "
        rf"& \multicolumn{{2}}{{c}}{{\textcolor{{blue}}{{\underline{{{second_counts['CM']}}}}}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results/table_unimodal_comparison.tex",
        help="Output .tex file path",
    )
    parser.add_argument(
        "--temporal_csv",
        type=Path,
        default=RESULT_CSVS["T"],
        help="CSV for temporal unimodal probe",
    )
    parser.add_argument(
        "--visual_csv",
        type=Path,
        default=RESULT_CSVS["V"],
        help="CSV for visual unimodal probe",
    )
    parser.add_argument(
        "--multimodal_csv",
        type=Path,
        default=RESULT_CSVS["CM"],
        help="CSV for contrastive multimodal probe",
    )
    args = parser.parse_args()

    t_data  = load_csv(args.temporal_csv)
    v_data  = load_csv(args.visual_csv)
    cm_data = load_csv(args.multimodal_csv)

    table = build_table(t_data, v_data, cm_data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + "\n")
    print(f"Table written to: {args.out}")


if __name__ == "__main__":
    main()
