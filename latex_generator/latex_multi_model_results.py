"""Generate a LaTeX table comparing multiple model result files.

All runtime settings are loaded from the YAML configuration defined by
`DEFAULT_CONFIG_PATH`.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml

DEFAULT_DECIMALS = 5


def resolve_path(path: str) -> str:
    """Return an absolute path for local or relative references."""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(__file__).parent / candidate).resolve())


# Default config path relative to this script
DEFAULT_CONFIG_PATH = resolve_path("../src/configs/multi_model_config.yaml")


def clean_name(path: str) -> str:
    filename = path.replace("\\", "/").split("/")[-1]
    name = re.sub(r"\.(csv|txt|npz)$", "", filename, flags=re.IGNORECASE)
    lowered = name.lower()

    if "behind_electricity" in lowered:
        return "Electricity (behind)"
    if "middle_electricity" in lowered:
        return "Electricity (middle)"
    if lowered == "electricity":
        return "Electricity"
    if "exchange" in lowered:
        return "Exchange"
    if "solar" in lowered or "solar_al" in lowered:
        return "Solar"
    if "pems04" in lowered or "traffic" in lowered:
        return "Traffic"
    if "ettm1" in lowered:
        return "ETTm1"
    if "ettm2" in lowered:
        return "ETTm2"
    if "etth1" in lowered:
        return "ETTh1"
    if "etth2" in lowered:
        return "ETTh2"
    if "weather" in lowered:
        return "Weather"

    return name


def format_metric(value: float | None, decimals: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def load_configuration(config_path: str) -> Dict:
    resolved = resolve_path(config_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Could not locate config file: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_model_entries(entries: List[Dict]) -> Tuple[List[str], List[str]]:
    if not entries:
        raise ValueError("Configuration must include at least one model entry.")

    result_files: List[str] = []
    model_names: List[str] = []

    for idx, entry in enumerate(entries):
        if "file" not in entry or "name" not in entry:
            raise ValueError(
                f"Model entry #{idx + 1} must define both 'file' and 'name'."
            )
        result_files.append(entry["file"])
        model_names.append(entry["name"])

    return result_files, model_names


def generate_latex_table(
    result_files: List[str],
    model_names: List[str],
    dataset_order: List[str],
    horizon_config: List[int] | None,
    decimals: int,
    output_dir: str,
    output_filename: str,
) -> Path:
    if len(result_files) != len(model_names):
        raise ValueError("Each result file must have a corresponding model name.")

    # Structure: dataset -> horizon -> model -> {MAE, MSE}
    per_horizon_metrics: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    allowed_horizons: set[int] | None = None
    if horizon_config is not None:
        allowed_horizons = {int(h) for h in horizon_config}

    for file_path, model_name in zip(result_files, model_names):
        csv_path = resolve_path(file_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not locate results file: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {"dataset_name", "horizon", "mae", "mse"}
        if not required_cols.issubset(df.columns):
            missing = ", ".join(sorted(required_cols - set(df.columns)))
            raise ValueError(
                f"File {csv_path} is missing required columns: {missing}"
            )

        df["dataset"] = df["dataset_name"].apply(clean_name)

        for _, row in df.iterrows():
            dataset = row["dataset"]
            horizon = int(row["horizon"])
            mae = float(row["mae"])
            mse = float(row["mse"])

            if allowed_horizons is not None and horizon not in allowed_horizons:
                continue

            per_horizon_metrics[dataset][horizon][model_name] = {"MAE": mae, "MSE": mse}

    if not per_horizon_metrics:
        raise RuntimeError("No data loaded; check the input files and columns.")

    # Compute mean metrics per dataset/model (average across horizons)
    dataset_means: Dict[str, Dict[str, Dict[str, float | None]]] = defaultdict(dict)
    for dataset, horizon_dict in per_horizon_metrics.items():
        for model in model_names:
            mae_vals = []
            mse_vals = []
            for horizon_data in horizon_dict.values():
                if model in horizon_data:
                    mae_vals.append(horizon_data[model]["MAE"])
                    mse_vals.append(horizon_data[model]["MSE"])
            mae_mean = float(sum(mae_vals) / len(mae_vals)) if mae_vals else None
            mse_mean = float(sum(mse_vals) / len(mse_vals)) if mse_vals else None
            dataset_means[dataset][model] = {"MAE": mae_mean, "MSE": mse_mean}

    if dataset_order:
        datasets: List[str] = [ds for ds in dataset_order if ds in per_horizon_metrics]
        remaining = sorted(ds for ds in per_horizon_metrics if ds not in dataset_order)
        datasets.extend(remaining)
    else:
        datasets = sorted(per_horizon_metrics.keys())

    # counters for best/second-best across all dataset/horizon/metric cells
    best_counts = {model: 0 for model in model_names}
    second_counts = {model: 0 for model in model_names}

    latex: List[str] = []
    latex.append(r"\begin{table*}[ht!]")
    latex.append(r"\centering")
    latex.append(r"\caption{The table presents the zero-shot evaluation of the CM-Mamba-mini model, which was trained using only the temporal and visual data, as well as the embeddings (concatenation) before the forecast head.}")
    latex.append(r"\small")
    latex.append(r"\begin{adjustbox}{max width=\textwidth}")
    latex.append(r"\label{tab:contrastive_multimodal_comparison}")
    
    # Build column specification: Models | Metric | Model1 MSE MAE | Model2 MSE MAE | ...
    num_models = len(model_names)
    col_spec = "l c" + " c c" * num_models
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"\toprule")

    # Create model-based headers
    header_parts = [r"\textbf{Models}", r"\textbf{Metric}"]
    for model in model_names:
        header_parts.append(rf"\multicolumn{{2}}{{c}}{{\textbf{{{model}}}}}")
    latex.append(" & ".join(header_parts) + r" \\")
    
    # Add cmidrules for each model column
    for i in range(num_models):
        col_start = 3 + i * 2
        col_end = col_start + 1
        latex.append(rf"\cmidrule(lr){{{col_start}-{col_end}}}")
    
    # Metric sub-header
    metric_parts = ["", ""]
    for _ in model_names:
        metric_parts.extend([r"\textbf{MSE}", r"\textbf{MAE}"])
    latex.append(" & ".join(metric_parts) + r" \\")
    latex.append(r"\midrule")

    # Fixed horizons for the template format
    fixed_horizons = [96, 192, 336, 720]
    
    for idx, dataset in enumerate(datasets):
        for h_idx, horizon in enumerate(fixed_horizons):
            row_parts: List[str] = []
            
            # First column: dataset name (only on first horizon row)
            if h_idx == 0:
                row_parts.append(rf"\multirow{{4}}{{*}}{{\textbf{{{dataset}}}}}")
            else:
                row_parts.append("")
            
            # Second column: horizon
            row_parts.append(str(horizon))
            
            # Collect all metrics for this dataset/horizon to find best and second best
            horizon_metrics: Dict[str, Dict[str, float | None]] = {}
            for model in model_names:
                mae_val = None
                mse_val = None
                
                if dataset in per_horizon_metrics and horizon in per_horizon_metrics[dataset]:
                    if model in per_horizon_metrics[dataset][horizon]:
                        mse_val = per_horizon_metrics[dataset][horizon][model]["MSE"]
                        mae_val = per_horizon_metrics[dataset][horizon][model]["MAE"]
                
                horizon_metrics[model] = {"MSE": mse_val, "MAE": mae_val}
            
            # Find best and second best for MSE and MAE
            rankings = find_best_results(horizon_metrics)
            
            # For each model, add MSE and MAE with formatting
            for model in model_names:
                mse_val = horizon_metrics[model]["MSE"]
                mae_val = horizon_metrics[model]["MAE"]
                
                row_parts.append(format_ranked_value(mse_val, model, "MSE", rankings, decimals))
                row_parts.append(format_ranked_value(mae_val, model, "MAE", rankings, decimals))

                # tally counts
                if rankings["MSE"].get("best") == model:
                    best_counts[model] += 1
                elif rankings["MSE"].get("second") == model:
                    second_counts[model] += 1
                if rankings["MAE"].get("best") == model:
                    best_counts[model] += 1
                elif rankings["MAE"].get("second") == model:
                    second_counts[model] += 1
            
            latex.append(" & ".join(row_parts) + r" \\")
        
        # Add horizontal line after each dataset (except last)
        if idx < len(datasets) - 1:
            latex.append(r"\midrule")

    # summary rows for counts
    latex.append(r"\midrule")

    def fmt_count(value: int, kind: str) -> str:
        if kind == "best":
            return rf"\textcolor{{red}}{{\textbf{{{value}}}}}"
        if kind == "second":
            return rf"\textcolor{{blue}}{{\underline{{{value}}}}}"
        return str(value)

    best_row = [r"\textbf{Best count}", ""]
    for model in model_names:
        best_row.append(fmt_count(best_counts.get(model, 0), "best"))
        best_row.append("")
    latex.append(" & ".join(best_row) + r" \\")

    second_row = [r"\textbf{Second-best count}", ""]
    for model in model_names:
        second_row.append(fmt_count(second_counts.get(model, 0), "second"))
        second_row.append("")
    latex.append(" & ".join(second_row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{adjustbox}")
    latex.append(r"\end{table*}")

    final_latex = "\n".join(latex)
    output_dir_path = Path(resolve_path(output_dir))
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / output_filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_latex)

    return out_path


def find_best_results(all_results: Dict[str, Dict[str, float | None]]) -> Dict[str, Dict[str, str | None]]:
    rankings = {metric: {"best": None, "second": None} for metric in ["MSE", "MAE"]}

    for metric in ["MSE", "MAE"]:
        valid_results = [
            (model, values.get(metric))
            for model, values in all_results.items()
            if values.get(metric) is not None
        ]
        if not valid_results:
            continue
        valid_results.sort(key=lambda x: x[1])
        rankings[metric]["best"] = valid_results[0][0]
        if len(valid_results) > 1:
            rankings[metric]["second"] = valid_results[1][0]

    return rankings


def format_ranked_value(
    value: float | None,
    model: str,
    metric: str,
    rankings: Dict[str, Dict[str, str | None]],
    decimals: int,
) -> str:
    if value is None:
        return "-"

    formatted = f"{value:.{decimals}f}"
    best = rankings[metric].get("best")
    second = rankings[metric].get("second")

    if model == best:
        return rf"\textcolor{{red}}{{\textbf{{{formatted}}}}}"
    if model == second:
        return rf"\textcolor{{blue}}{{\underline{{{formatted}}}}}"
    return formatted


def main() -> None:
    config = load_configuration(DEFAULT_CONFIG_PATH)
    result_files, model_names = parse_model_entries(config.get("models", []))

    dataset_order = config.get("dataset_order", [])
    horizons = config.get("horizons")
    decimals = int(config.get("decimals", DEFAULT_DECIMALS))
    output_cfg = config.get("output", {})
    output_dir = output_cfg.get("dir", "../results")
    output_filename = output_cfg.get("filename", "multi_model_comparison.tex")

    out_path = generate_latex_table(
        result_files,
        model_names,
        dataset_order,
        horizons,
        decimals,
        output_dir,
        output_filename,
    )

    print(f"Wrote comparison LaTeX table to: {out_path}")


if __name__ == "__main__":
    main()
