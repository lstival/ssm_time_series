import os
import re
from pathlib import Path

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Please install pyyaml to run this script: pip install pyyaml"
    ) from exc

# -----------------------
# 1. Config
# -----------------------
DEFAULT_CONFIG_PATH = Path(__file__).with_name("cm_mamba_runs.yaml")
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# -----------------------
# 2. Dataset name normalization
# -----------------------
target_datasets = [
    "ETTm1",
    "ETTm2",
    "ETTh1",
    "ETTh2",
    "Traffic",
    "Weather",
    "Exchange",
    "Solar",
    "Electricity",
]

def clean_name(path: str) -> str:
    filename = path.replace("\\", "/").split("/")[-1]
    name = re.sub(r"\.(csv|txt|npz)$", "", filename)

    lower = name.lower()
    if "ettm1" in lower:
        return "ETTm1"
    if "ettm2" in lower:
        return "ETTm2"
    if "etth1" in lower:
        return "ETTh1"
    if "etth2" in lower:
        return "ETTh2"
    if "traffic" in lower or "pems04" in lower:
        return "Traffic"
    if "weather" in lower:
        return "Weather"
    if "exchange" in lower or "exchange_rate" in lower:
        return "Exchange"
    if "electricity" in lower:
        return "Electricity"
    if "solar" in lower:
        return "Solar"

    return name


def compute_means_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV and return per-dataset mean of MSE/MAE for target datasets."""

    df = pd.read_csv(csv_path)
    required = {"dataset_name", "mae", "mse"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {csv_path}")

    df["dataset"] = df["dataset_name"].apply(clean_name)
    df_filtered = df[df["dataset"].isin(target_datasets)]

    return df_filtered.groupby("dataset")[['mae', 'mse']].mean().reset_index()

# -----------------------
# 3. Baseline results (hardcoded)
# -----------------------
baseline_data = {
    "ETTm1": {
        "LightGTS": (0.327, 0.370),
        "Timer": (0.768, 0.568),
        "MOIRAI": (0.390, 0.389),
        "Chronos": (0.551, 0.453),
        "TimesFM": (0.435, 0.418),
        "Time-MoE": (0.376, 0.406),
    },
    "ETTm2": {
        "LightGTS": (0.247, 0.316),
        "Timer": (0.315, 0.356),
        "MOIRAI": (0.276, 0.320),
        "Chronos": (0.293, 0.331),
        "TimesFM": (0.347, 0.360),
        "Time-MoE": (0.315, 0.365),
    },
    "ETTh1": {
        "LightGTS": (0.388, 0.419),
        "Timer": (0.562, 0.483),
        "MOIRAI": (0.510, 0.469),
        "Chronos": (0.533, 0.452),
        "TimesFM": (0.479, 0.442),
        "Time-MoE": (0.394, 0.420),
    },
    "ETTh2": {
        "LightGTS": (0.348, 0.395),
        "Timer": (0.370, 0.400),
        "MOIRAI": (0.354, 0.377),
        "Chronos": (0.392, 0.397),
        "TimesFM": (0.400, 0.403),
        "Time-MoE": (0.403, 0.415),
    },
    "Traffic": {
        "LightGTS": (0.561, 0.381),
        "Timer": (0.613, 0.407),
        "MOIRAI": (None, None),
        "Chronos": (0.615, 0.421),
        "TimesFM": (None, None),
        "Time-MoE": (None, None),
    },
    "Weather": {
        "LightGTS": (0.208, 0.256),
        "Timer": (0.292, 0.313),
        "MOIRAI": (0.260, 0.275),
        "Chronos": (0.288, 0.309),
        "TimesFM": (None, None),
        "Time-MoE": (0.270, 0.300),
    },
    "Exchange": {
        "LightGTS": (0.347, 0.396),
        "Timer": (0.392, 0.425),
        "MOIRAI": (0.385, 0.417),
        "Chronos": (0.370, 0.412),
        "TimesFM": (0.390, 0.417),
        "Time-MoE": (0.432, 0.454),
    },
    "Solar": {
        "LightGTS": (0.191, 0.271),
        "Timer": (0.771, 0.604),
        "MOIRAI": (0.714, 0.704),
        "Chronos": (0.393, 0.319),
        "TimesFM": (0.500, 0.397),
        "Time-MoE": (0.411, 0.428),
    },
    "Electricity": {
        "LightGTS": (0.213, 0.308),
        "Timer": (0.297, 0.375),
        "MOIRAI": (0.188, 0.273),
        "Chronos": (None, None),
        "TimesFM": (None, None),
        "Time-MoE": (None, None),
    },
}

BASELINE_MODELS = [
    ("LightGTS", "LightGTS (2025)"),
    ("Timer", "Timer (2024)"),
    ("MOIRAI", "MOIRAI (2024)"),
    ("Chronos", "Chronos (2024)"),
    ("TimesFM", "TimesFM (2024)"),
    ("Time-MoE", "Time-MoE (2025)"),
]

# -----------------------
# 4. Helpers
# -----------------------
def slugify_model_name(name: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or fallback


def get_metric(means_df: pd.DataFrame, dataset: str, metric: str):
    if means_df is None:
        return None
    match = means_df[means_df["dataset"] == dataset]
    if match.empty:
        return None
    return float(match[metric].iloc[0])


def find_best_results(all_results):
    """Find best and second best results for MSE and MAE separately."""
    rankings = {metric: {"best": None, "second": None} for metric in ["MSE", "MAE"]}

    for metric in ["MSE", "MAE"]:
        valid_results = [
            (model, values[metric])
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


def format_value(value, model, metric, rankings):
    """Format value with colors and styles based on ranking."""
    if value is None or pd.isna(value):
        return "-"

    formatted = f"{value:.3f}"
    best_model = rankings[metric].get("best")
    second_model = rankings[metric].get("second")

    if best_model == model:
        return f"\\textcolor{{red}}{{\\textbf{{{formatted}}}}}"
    if second_model == model:
        return f"\\textcolor{{blue}}{{\\underline{{{formatted}}}}}"
    return formatted


def load_config():
    config_path = Path(os.environ.get("CM_MAMBA_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Create it or set CM_MAMBA_CONFIG to the correct path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config_path, config


def build_cm_mamba_runs(config, results_dir: Path):
    variants = config.get("cm_mamba_variants", [])
    if not variants:
        raise ValueError("No CM-Mamba variants found in config under 'cm_mamba_variants'.")

    runs = []
    for idx, variant in enumerate(variants):
        display = variant.get("display_name") or f"CM-Mamba {idx + 1}"
        variant_key = slugify_model_name(
            variant.get("id") or display, fallback=f"cm_mamba_{idx + 1}"
        )

        csv_entry = variant.get("file") or variant.get("csv") or variant.get("path")
        if not csv_entry:
            raise ValueError(
                f"Variant '{display}' is missing a CSV path. Use key 'file' in the config."
            )

        csv_path = Path(csv_entry)
        if not csv_path.is_absolute():
            csv_path = results_dir / csv_path
        csv_path = csv_path.resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for variant '{display}': {csv_path}")

        means_df = compute_means_from_csv(csv_path)

        runs.append(
            {
                "key": variant_key,
                "display": display,
                "csv_path": csv_path,
                "means": means_df,
            }
        )

    return runs


def build_latex(table_models, cm_mamba_runs, baseline_models):
    latex = []
    best_counts = {model: 0 for model, _ in table_models}
    second_counts = {model: 0 for model, _ in table_models}

    latex.append(r"\begin{table*}[ht]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Full results of zero-shot forecasting experiments. The average results of all predicted lengths are listed. Lower MSE or MAE indicate better predictions. Red: the best, Blue: the 2nd best.}"
    )
    latex.append(r"\renewcommand{\arraystretch}{1.2}")
    latex.append(r"\setlength{\tabcolsep}{4pt}")
    latex.append(r"\begin{adjustbox}{max width=\textwidth}")
    num_model_columns = len(table_models) * 2
    tabular_spec = "l" + "c" * num_model_columns
    latex.append("\\begin{tabular}{%s}" % tabular_spec)
    latex.append(r"\toprule")

    latex.append(rf"\multicolumn{{1}}{{c}}{{}} & \multicolumn{{{num_model_columns}}}{{c}}{{Models}} \\")
    latex.append(rf"\cmidrule(lr){{2-{num_model_columns + 1}}}")

    model_headers = " & ".join(
        [rf"\multicolumn{{2}}{{c}}{{\textbf{{{display}}}}}" for _, display in table_models]
    )
    latex.append(rf"\textbf{{Metric}} & {model_headers} \\")

    metric_row = " & ".join(["MSE", "MAE"] * len(table_models))
    latex.append(rf" & {metric_row} \\")
    latex.append(r"\midrule")

    for dataset in target_datasets:
        dataset_results = {}
        row_parts = [dataset]

        for run in cm_mamba_runs:
            mse_val = get_metric(run["means"], dataset, "mse")
            mae_val = get_metric(run["means"], dataset, "mae")
            dataset_results[run["key"]] = {"MSE": mse_val, "MAE": mae_val}

        for model in baseline_models:
            mse, mae = baseline_data.get(dataset, {}).get(model, (None, None))
            dataset_results[model] = {"MSE": mse, "MAE": mae}

        rankings = find_best_results(dataset_results)

        # tally best/second placements for counts row
        for model_name, values in dataset_results.items():
            for metric in ["MSE", "MAE"]:
                val = values.get(metric)
                if val is None:
                    continue
                if rankings[metric].get("best") == model_name:
                    best_counts[model_name] += 1
                elif rankings[metric].get("second") == model_name:
                    second_counts[model_name] += 1

        for run in cm_mamba_runs:
            row_parts.append(
                format_value(get_metric(run["means"], dataset, "mse"), run["key"], "MSE", rankings)
            )
            row_parts.append(
                format_value(get_metric(run["means"], dataset, "mae"), run["key"], "MAE", rankings)
            )

        for model in baseline_models:
            mse, mae = baseline_data.get(dataset, {}).get(model, (None, None))
            row_parts.append(format_value(mse, model, "MSE", rankings))
            row_parts.append(format_value(mae, model, "MAE", rankings))

        latex.append(" & ".join(row_parts) + r" \\")

    # Summary rows for counts of best/second-best placements
    latex.append(r"\midrule")

    def fmt_count(value: int, kind: str) -> str:
        if value is None:
            return "-"
        if kind == "best":
            return f"\\textcolor{{red}}{{\\textbf{{{value}}}}}"
        if kind == "second":
            return f"\\textcolor{{blue}}{{\\underline{{{value}}}}}"
        return str(value)

    # Best counts row
    best_row_parts = ["Best count"]
    for model, _ in table_models:
        count = best_counts.get(model, 0)
        best_row_parts.append(fmt_count(count, "best"))
        best_row_parts.append("")  # keep two columns per model
    latex.append(" & ".join(best_row_parts) + r" \\")

    # Second-best counts row
    second_row_parts = ["Second-best count"]
    for model, _ in table_models:
        count = second_counts.get(model, 0)
        second_row_parts.append(fmt_count(count, "second"))
        second_row_parts.append("")
    latex.append(" & ".join(second_row_parts) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{adjustbox}")
    latex.append(r"\label{tab:comparison_mean}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def write_output(final_latex: str, config: dict, config_path: Path) -> Path:
    out_dir = Path(config.get("output_dir", config_path.parent)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"average_results_comparison_{config_path.stem}.tex"
    out_path = out_dir / out_filename

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_latex)

    return out_path


def main():
    config_path, config = load_config()
    results_dir = Path(config.get("results_dir", DEFAULT_RESULTS_DIR)).resolve()
    cm_mamba_runs = build_cm_mamba_runs(config, results_dir)

    baseline_models = [model for model, _ in BASELINE_MODELS]
    table_models = [(run["key"], run["display"]) for run in cm_mamba_runs] + BASELINE_MODELS

    final_latex = build_latex(table_models, cm_mamba_runs, baseline_models)
    out_path = write_output(final_latex, config, config_path)

    print(f"Wrote LaTeX comparison table to: {out_path}")
    print("\nDataset mean values per CM-Mamba variant:")
    for run in cm_mamba_runs:
        print(f"\n{run['display']} (key={run['key']}):")
        for _, row in run["means"].iterrows():
            print(f"{row['dataset']}: MSE={row['mse']:.3f}, MAE={row['mae']:.3f}")

if __name__ == "__main__":
    main()
