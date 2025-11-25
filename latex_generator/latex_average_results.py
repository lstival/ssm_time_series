import pandas as pd
import re
import os
import numpy as np

# -----------------------
# 1. Load CSV with metrics
# -----------------------
file_name = r"C:\WUR\ssm_time_series\results\icml_zeroshot_forecast_20251121_1108.csv"
df = pd.read_csv(file_name)

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
    if "traffic" in name.lower():
        return "Traffic"
    if "weather" in name.lower():
        return "Weather"
    if "exchange" in name.lower() or "exchange_rate" in name.lower():
        return "Exchange"
    if "electricity" in name.lower():
        return "Electricity"
    
    return name

df["dataset"] = df["dataset_name"].apply(clean_name)

# -----------------------
# 3. Filter for datasets in average_result.tex and calculate means
# -----------------------
target_datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Traffic", "Weather", "Exchange", "Electricity"]
df_filtered = df[df["dataset"].isin(target_datasets)]

# Group by dataset and calculate mean across all horizons
df_means = df_filtered.groupby("dataset")[["mae", "mse"]].mean().reset_index()

# -----------------------
# 4. Baseline results from average_result.tex (hardcoded)
# -----------------------
baseline_data = {
    "ETTm1": {
        "PDF": (0.342, 0.376), "iTransformer": (0.347, 0.378), "Pathformer": (0.357, 0.375),
        "FITS": (0.357, 0.377), "TimeMixer": (0.356, 0.380), "PatchTST": (0.349, 0.381)
    },
    "ETTm2": {
        "PDF": (0.250, 0.313), "iTransformer": (0.258, 0.318), "Pathformer": (0.253, 0.309),
        "FITS": (0.254, 0.313), "TimeMixer": (0.257, 0.318), "PatchTST": (0.256, 0.314)
    },
    "ETTh1": {
        "PDF": (0.407, 0.426), "iTransformer": (0.440, 0.445), "Pathformer": (0.417, 0.426),
        "FITS": (0.408, 0.427), "TimeMixer": (0.427, 0.441), "PatchTST": (0.419, 0.436)
    },
    "ETTh2": {
        "PDF": (0.347, 0.391), "iTransformer": (0.359, 0.396), "Pathformer": (0.360, 0.395),
        "FITS": (0.335, 0.386), "TimeMixer": (0.347, 0.394), "PatchTST": (0.351, 0.395)
    },
    "Traffic": {
        "PDF": (0.395, 0.270), "iTransformer": (0.397, 0.281), "Pathformer": (0.416, 0.264),
        "FITS": (0.429, 0.302), "TimeMixer": (0.410, 0.279), "PatchTST": (0.397, 0.275)
    },
    "Weather": {
        "PDF": (0.227, 0.263), "iTransformer": (0.232, 0.270), "Pathformer": (0.225, 0.258),
        "FITS": (0.244, 0.281), "TimeMixer": (0.225, 0.263), "PatchTST": (0.224, 0.261)
    },
    "Exchange": {
        "PDF": (0.350, 0.397), "iTransformer": (0.321, 0.384), "Pathformer": (0.384, 0.414),
        "FITS": (0.349, 0.396), "TimeMixer": (0.385, 0.418), "PatchTST": (0.322, 0.385)
    },
    "Electricity": {
        "PDF": (0.160, 0.253), "iTransformer": (0.163, 0.258), "Pathformer": (0.168, 0.261),
        "FITS": (0.169, 0.265), "TimeMixer": (0.185, 0.284), "PatchTST": (0.171, 0.270)
    }
}

models = ["PDF", "iTransformer", "Pathformer", "FITS", "TimeMixer", "PatchTST"]

# -----------------------
# 5. Find best and second best for formatting
# -----------------------
def find_best_results(dataset_results, our_result):
    """Find best and second best results for MSE and MAE separately"""
    all_results = {}
    
    # Add baseline results
    for model in models:
        if model in dataset_results:
            mse, mae = dataset_results[model]
            all_results[model] = {"MSE": mse, "MAE": mae}
    
    # Add our result
    our_mse, our_mae = our_result
    all_results["Our"] = {"MSE": our_mse, "MAE": our_mae}
    
    # Find rankings for MSE and MAE
    rankings = {"MSE": {}, "MAE": {}}
    
    for metric in ["MSE", "MAE"]:
        sorted_results = sorted(all_results.items(), key=lambda x: x[1][metric])
        rankings[metric]["best"] = sorted_results[0][0]
        rankings[metric]["second"] = sorted_results[1][0]
    
    return rankings

def format_value(value, model, metric, rankings):
    """Format value with colors and styles based on ranking"""
    formatted = f"{value:.3f}"
    
    if rankings[metric]["best"] == model:
        return f"\\textcolor{{red}}{{\\textbf{{{formatted}}}}}"
    elif rankings[metric]["second"] == model:
        return f"\\textcolor{{blue}}{{\\underline{{{formatted}}}}}"
    else:
        return formatted

# -----------------------
# 6. Build LaTeX table
# -----------------------
latex = []

latex.append(r"\begin{table*}[ht]")
latex.append(r"\centering")
latex.append(r"\caption{Performance comparison (MSE / MAE) across datasets and models - Mean across all forecast horizons.}")
latex.append(r"\renewcommand{\arraystretch}{1.2}")
latex.append(r"\setlength{\tabcolsep}{4pt}")
latex.append(r"\begin{adjustbox}{max width=\textwidth}")
latex.append(r"\begin{tabular}{lcccccccccccccc}")
latex.append(r"\toprule")

# Header
latex.append(r"\multirow{2}{*}{Dataset} & ")
latex.append(r"\multicolumn{2}{c}{\textbf{Our}} & ")
latex.append(r"\multicolumn{2}{c}{\textbf{PDF (2024)}} & ")
latex.append(r"\multicolumn{2}{c}{\textbf{iTransformer (2024)}} & ")
latex.append(r"\multicolumn{2}{c}{\textbf{Pathformer (2024)}} & ")
latex.append(r"\multicolumn{2}{c}{\textbf{FITS (2024)}} & ")
latex.append(r"\multicolumn{2}{c}{\textbf{TimeMixer (2024)}} & ")
latex.append(r"\multicolumn{2}{c}{\textbf{PatchTST (2023)}} \\")

latex.append(r" & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\")
latex.append(r"\midrule")

# Data rows
for dataset in target_datasets:
    if dataset in df_means["dataset"].values:
        our_data = df_means[df_means["dataset"] == dataset].iloc[0]
        our_mse = our_data["mse"]
        our_mae = our_data["mae"]
    else:
        our_mse = 0.0
        our_mae = 0.0
    
    # Get rankings for this dataset
    rankings = find_best_results(baseline_data.get(dataset, {}), (our_mse, our_mae))
    
    row_parts = [dataset]
    
    # Add our results first
    row_parts.append(format_value(our_mse, "Our", "MSE", rankings))
    row_parts.append(format_value(our_mae, "Our", "MAE", rankings))
    
    # Add baseline results
    for model in models:
        if dataset in baseline_data and model in baseline_data[dataset]:
            mse, mae = baseline_data[dataset][model]
            row_parts.append(format_value(mse, model, "MSE", rankings))
            row_parts.append(format_value(mae, model, "MAE", rankings))
        else:
            row_parts.extend(["-", "-"])
    
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

out_filename = "average_results_comparison.tex"
out_dir = os.path.dirname(file_name) or os.getcwd()
out_path = os.path.join(out_dir, out_filename)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(final_latex)

print(f"Wrote LaTeX comparison table to: {out_path}")

# Print summary
print("\nDataset mean values (Our model):")
for _, row in df_means.iterrows():
    print(f"{row['dataset']}: MSE={row['mse']:.3f}, MAE={row['mae']:.3f}")