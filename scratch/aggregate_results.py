import os
import pandas as pd
from pathlib import Path

ROOT_DIR = Path("/home/WUR/stiva001/WUR/ssm_time_series")
GRID_DIR = ROOT_DIR / "results" / "mop_grid_clip_v1"

experiments = [
    "baseline", "revin", "minmax", "mlp", "residual", "ln_head", 
    "scale", "temp_0.5", "temp_2.0", "scale_cond", "revin_mlp"
]

all_results = []

for exp in experiments:
    exp_dir = GRID_DIR / exp
    
    # Zero-shot
    zs_file = exp_dir / "eval_zeroshot" / "eval_mop_flex.csv"
    if zs_file.exists():
        df = pd.read_csv(zs_file)
        # Average MSE/MAE across all datasets and horizons
        avg_mse = df['mse'].mean()
        avg_mae = df['mae'].mean()
        all_results.append({
            "experiment": exp,
            "type": "zeroshot",
            "avg_mse": avg_mse,
            "avg_mae": avg_mae
        })
    
    # Few-shot
    for ds in ["weather.csv", "traffic.csv", "electricity.csv"]:
        fs_file = exp_dir / f"eval_fewshot_{ds}" / "eval_mop_flex.csv"
        if fs_file.exists():
            df = pd.read_csv(fs_file)
            avg_mse = df['mse'].mean()
            avg_mae = df['mae'].mean()
            all_results.append({
                "experiment": exp,
                "type": f"fewshot_{ds}",
                "avg_mse": avg_mse,
                "avg_mae": avg_mae
            })

results_df = pd.DataFrame(all_results)
print(results_df.to_string())

# Pivot table for better viewing
if not results_df.empty:
    pivot = results_df.pivot(index="experiment", columns="type", values="avg_mse")
    print("\nPivot Table (MSE):")
    print(pivot.to_string())
    
    # Find top 2 based on overall few-shot performance or zero-shot?
    # Usually we look at zero-shot for generalizability, but few-shot for target datasets.
    # Let's calculate an overall score (mean of all avg_mse)
    summary = results_df.groupby("experiment")["avg_mse"].mean().sort_values()
    print("\nSummary (Mean MSE across all scenarios):")
    print(summary)
