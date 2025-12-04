import pandas as pd
import re
import os

# -----------------------
# 1. Load CSV with metrics
# -----------------------
file_name = "chronos_supervised_zeroshot_20251202_0931.csv"
file_path = f"../results/{file_name}"
df = pd.read_csv(file_path)

# -----------------------
# 2. Normalize dataset names to match average_result.tex
# -----------------------
def clean_name(path):
    filename = path.replace("\\", "/").split("/")[-1]
    name = re.sub(r"\.(csv|txt|npz)$", "", filename)
    
    # Map to names used in average_result.tex
    if "ettm1" in name.lower():
        return "ETTm1"
    if "ettm2" in name.lower():
        return "ETTm2"
    if "etth1" in name.lower():
        return "ETTh1"
    if "etth2" in name.lower():
        return "ETTh2"
    if "traffic" in name.lower() or "PEMS04".lower() in name.lower():
        return "Traffic"
    if "weather" in name.lower():
        return "Weather"
    if "exchange" in name.lower() or "exchange_rate" in name.lower():
        return "Exchange"
    if "electricity" in name.lower():
        return "Electricity"
    if "solar" in name.lower():
        return "Solar"
    
    return name

df["dataset"] = df["dataset_name"].apply(clean_name)

# -----------------------
# 3. Filter for datasets in average_result.tex and calculate means
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
df_filtered = df[df["dataset"].isin(target_datasets)]

# Group by dataset and calculate mean across all horizons
df_means = df_filtered.groupby("dataset")[["mae", "mse"]].mean().reset_index()

# -----------------------
# 4. Baseline results (hardcoded)
# -----------------------
baseline_data = {
    "ETTm1": {
        "LightGTS-mini": (0.327, 0.370),
        "Timer": (0.768, 0.568),
        "MOIRAI": (0.390, 0.389),
        "Chronos": (0.551, 0.453),
        "TimesFM": (0.435, 0.418),
        "Time-MoE": (0.376, 0.406),
    },
    "ETTm2": {
        "LightGTS-mini": (0.247, 0.316),
        "Timer": (0.315, 0.356),
        "MOIRAI": (0.276, 0.320),
        "Chronos": (0.293, 0.331),
        "TimesFM": (0.347, 0.360),
        "Time-MoE": (0.315, 0.365),
    },
    "ETTh1": {
        "LightGTS-mini": (0.388, 0.419),
        "Timer": (0.562, 0.483),
        "MOIRAI": (0.510, 0.469),
        "Chronos": (0.533, 0.452),
        "TimesFM": (0.479, 0.442),
        "Time-MoE": (0.394, 0.420),
    },
    "ETTh2": {
        "LightGTS-mini": (0.348, 0.395),
        "Timer": (0.370, 0.400),
        "MOIRAI": (0.354, 0.377),
        "Chronos": (0.392, 0.397),
        "TimesFM": (0.400, 0.403),
        "Time-MoE": (0.403, 0.415),
    },
    "Traffic": {
        "LightGTS-mini": (0.561, 0.381),
        "Timer": (0.613, 0.407),
        "MOIRAI": (None, None),
        "Chronos": (0.615, 0.421),
        "TimesFM": (None, None),
        "Time-MoE": (None, None),
    },
    "Weather": {
        "LightGTS-mini": (0.208, 0.256),
        "Timer": (0.292, 0.313),
        "MOIRAI": (0.260, 0.275),
        "Chronos": (0.288, 0.309),
        "TimesFM": (None, None),
        "Time-MoE": (0.270, 0.300),
    },
    "Exchange": {
        "LightGTS-mini": (0.347, 0.396),
        "Timer": (0.392, 0.425),
        "MOIRAI": (0.385, 0.417),
        "Chronos": (0.370, 0.412),
        "TimesFM": (0.390, 0.417),
        "Time-MoE": (0.432, 0.454),
    },
    "Solar": {
        "LightGTS-mini": (0.191, 0.271),
        "Timer": (0.771, 0.604),
        "MOIRAI": (0.714, 0.704),
        "Chronos": (0.393, 0.319),
        "TimesFM": (0.500, 0.397),
        "Time-MoE": (0.411, 0.428),
    },
    "Electricity": {
        "LightGTS-mini": (0.213, 0.308),
        "Timer": (0.297, 0.375),
        "MOIRAI": (0.188, 0.273),
        "Chronos": (None, None),
        "TimesFM": (None, None),
        "Time-MoE": (None, None),
    },
}

table_models = [
    ("Our", "Our (Mean)"),
    ("LightGTS-mini", "LightGTS-mini"),
    ("Timer", "Timer (2024)"),
    ("MOIRAI", "MOIRAI (2024)"),
    ("Chronos", "Chronos (2024)"),
    ("TimesFM", "TimesFM (2024)"),
    ("Time-MoE", "Time-MoE (2025)"),
]

baseline_models = [model for model, _ in table_models if model != "Our"]

# -----------------------
# 5. Find best and second best for formatting
# -----------------------
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
    """Format value with colors and styles based on ranking"""
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

# -----------------------
# 6. Build LaTeX table
# -----------------------
latex = []

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

# Data rows
for dataset in target_datasets:
    if dataset in df_means["dataset"].values:
        our_data = df_means[df_means["dataset"] == dataset].iloc[0]
        our_mse = our_data["mse"]
        our_mae = our_data["mae"]
    else:
        our_mse = None
        our_mae = None

    dataset_results = {}
    if our_mse is not None and our_mae is not None:
        dataset_results["Our"] = {"MSE": our_mse, "MAE": our_mae}

    for model in baseline_models:
        mse, mae = baseline_data.get(dataset, {}).get(model, (None, None))
        dataset_results[model] = {"MSE": mse, "MAE": mae}

    rankings = find_best_results(dataset_results)

    row_parts = [dataset]
    row_parts.append(format_value(our_mse, "Our", "MSE", rankings))
    row_parts.append(format_value(our_mae, "Our", "MAE", rankings))

    for model in baseline_models:
        mse, mae = baseline_data.get(dataset, {}).get(model, (None, None))
        row_parts.append(format_value(mse, model, "MSE", rankings))
        row_parts.append(format_value(mae, model, "MAE", rankings))

    latex.append(" & ".join(row_parts) + r" \\")

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{adjustbox}")
latex.append(r"\label{tab:comparison_mean}")
latex.append(r"\end{table*}")

# -----------------------
# 7. Save LaTeX file
# -----------------------
final_latex = "\n".join(latex)

out_filename = f"average_results_comparison.tex_{file_name.replace(".csv", "")}.tex"
out_dir = os.path.dirname(file_name) or os.getcwd()
out_path = os.path.join(out_dir, out_filename)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(final_latex)

print(f"Wrote LaTeX comparison table to: {out_path}")

# Print summary
print("\nDataset mean values (Our model):")
for _, row in df_means.iterrows():
    print(f"{row['dataset']}: MSE={row['mse']:.3f}, MAE={row['mae']:.3f}")