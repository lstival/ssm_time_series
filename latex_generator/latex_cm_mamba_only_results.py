import os
import re
from pathlib import Path
import math

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

DEFAULT_DATASETS = [
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

DEFAULT_CAPTION = (
    "Zero-shot forecasting results with reduced comparison. "
    "Lower MSE or MAE indicate better performance. Red: best, Blue: second best."
)
DEFAULT_LABEL = "tab:comparison_cm_mamba_only"


# -----------------------
# 2. Dataset name normalization
# -----------------------
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


def compute_means_from_csv(csv_path: Path, target_datasets: list[str]) -> pd.DataFrame:
    """Load a CSV and return per-dataset mean of MSE/MAE for target datasets."""

    df = pd.read_csv(csv_path)
    required = {"dataset_name", "mae", "mse"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns {sorted(missing)} in {csv_path}")

    df["dataset"] = df["dataset_name"].apply(clean_name)
    df_filtered = df[df["dataset"].isin(target_datasets)]

    return df_filtered.groupby("dataset")[["mae", "mse"]].mean().reset_index()


# -----------------------
# 3. Helpers
# -----------------------
def slugify_model_name(name: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or fallback


def get_metric(means_df: pd.DataFrame | None, dataset: str, metric: str) -> float | None:
    if means_df is None:
        return None
    match = means_df[means_df["dataset"] == dataset]
    if match.empty:
        return None
    return float(match[metric].iloc[0]) 


def find_best_results(all_results: dict) -> dict:
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

        # Sort by score (ascending), then by model name for deterministic tie-breaking
        valid_results.sort(key=lambda x: (x[1], str(x[0])))

        best_model, best_value = valid_results[0]
        rankings[metric]["best"] = best_model

        # Second-best is the next *distinct* value (avoid coloring ties as second-best)
        second_model = None
        for model, value in valid_results[1:]:
            if not math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12):
                second_model = model
                break
        rankings[metric]["second"] = second_model

    return rankings


def format_value(value: float | None, model: str, metric: str, rankings: dict) -> str:
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


def load_config() -> tuple[Path, dict]:
    config_path = Path(os.environ.get("CM_MAMBA_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Create it or set CM_MAMBA_CONFIG to the correct path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config_path, config


def build_cm_mamba_runs(config: dict, results_dir: Path, target_datasets: list[str]) -> list[dict]:
    variants = config.get("cm_mamba_variants", [])
    if not variants:
        raise ValueError("No CM-Mamba variants found in config under 'cm_mamba_variants'.")

    runs: list[dict] = []
    used_keys: set[str] = set()
    for idx, variant in enumerate(variants):
        display = variant.get("display_name") or f"CM-Mamba {idx + 1}"
        base_key = slugify_model_name(
            variant.get("id") or display, fallback=f"cm_mamba_{idx + 1}"
        )
        variant_key = base_key
        # Ensure keys are unique; duplicate keys cause multiple columns to be colored as best/second.
        suffix = 2
        while variant_key in used_keys:
            variant_key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(variant_key)

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

        means_df = compute_means_from_csv(csv_path, target_datasets)

        runs.append(
            {
                "key": variant_key,
                "display": display,
                "csv_path": csv_path,
                "means": means_df,
            }
        )

    return runs


def build_latex_cm_only(
    cm_mamba_runs: list[dict],
    target_datasets: list[str],
    caption: str,
    label: str,
) -> str:
    latex: list[str] = []

    best_counts = {run["key"]: 0 for run in cm_mamba_runs}
    second_counts = {run["key"]: 0 for run in cm_mamba_runs}

    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(rf"\caption{{{caption}}}")
    latex.append(r"\renewcommand{\arraystretch}{1.2}")
    latex.append(r"\setlength{\tabcolsep}{6pt}")
    latex.append(r"\begin{adjustbox}{max width=0.4\textwidth}")

    num_model_columns = len(cm_mamba_runs) * 2
    tabular_spec = "l" + "c" * num_model_columns
    latex.append(rf"\begin{{tabular}}{{{tabular_spec}}}")
    latex.append(r"\toprule")

    latex.append(
        rf" & \multicolumn{{{num_model_columns}}}{{c}}{{\textbf{{Models}}}} \\")
    latex.append(rf"\cmidrule(lr){{2-{num_model_columns + 1}}}")

    model_headers = " & ".join(
        [rf"\multicolumn{{2}}{{c}}{{\textbf{{{run['display']}}}}}" for run in cm_mamba_runs]
    )
    latex.append(rf"\textbf{{Dataset}} & {model_headers} \\")

    metric_row = " & ".join(["MSE", "MAE"] * len(cm_mamba_runs))
    latex.append(rf" & {metric_row} \\")
    latex.append(r"\midrule")

    for dataset in target_datasets:
        dataset_results: dict[str, dict[str, float | None]] = {}
        row_parts: list[str] = [dataset]

        for run in cm_mamba_runs:
            mse_val = get_metric(run["means"], dataset, "mse")
            mae_val = get_metric(run["means"], dataset, "mae")
            dataset_results[run["key"]] = {"MSE": mse_val, "MAE": mae_val}

        rankings = find_best_results(dataset_results)

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
                format_value(dataset_results[run["key"]]["MSE"], run["key"], "MSE", rankings)
            )
            row_parts.append(
                format_value(dataset_results[run["key"]]["MAE"], run["key"], "MAE", rankings)
            )

        latex.append(" & ".join(row_parts) + r" \\")

    latex.append(r"\midrule")

    def fmt_count(value: int, kind: str) -> str:
        if kind == "best":
            return f"\\textcolor{{red}}{{\\textbf{{{value}}}}}"
        if kind == "second":
            return f"\\textcolor{{blue}}{{\\underline{{{value}}}}}"
        return str(value)

    best_row_parts = ["Best count"]
    for run in cm_mamba_runs:
        best_row_parts.append(fmt_count(best_counts.get(run["key"], 0), "best"))
        best_row_parts.append("")
    latex.append(" & ".join(best_row_parts) + r" \\")

    second_row_parts = ["Second-best count"]
    for run in cm_mamba_runs:
        second_row_parts.append(fmt_count(second_counts.get(run["key"], 0), "second"))
        second_row_parts.append("")
    latex.append(" & ".join(second_row_parts) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{adjustbox}")
    latex.append(rf"\label{{{label}}}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def write_output(final_latex: str, config: dict, config_path: Path) -> Path:
    out_dir = Path(config.get("output_dir", config_path.parent)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"cm_mamba_only_average_results_{config_path.stem}.tex"
    out_path = out_dir / out_filename

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_latex)

    return out_path


def main() -> None:
    config_path, config = load_config()

    results_dir = Path(config.get("results_dir", DEFAULT_RESULTS_DIR)).resolve()
    target_datasets = config.get("datasets") or DEFAULT_DATASETS
    caption = config.get("caption") or DEFAULT_CAPTION
    label = config.get("label") or DEFAULT_LABEL

    cm_mamba_runs = build_cm_mamba_runs(config, results_dir, target_datasets)

    final_latex = build_latex_cm_only(
        cm_mamba_runs=cm_mamba_runs,
        target_datasets=target_datasets,
        caption=caption,
        label=label,
    )
    out_path = write_output(final_latex, config, config_path)

    print(f"Wrote LaTeX comparison table to: {out_path}")


if __name__ == "__main__":
    main()
