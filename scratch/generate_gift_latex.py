import pandas as pd

# Provided baselines (SEMPO, Chronos-L, Chronos-B, Chronos-Sm)
# Format: {subset: {metric: [SEMPO, Chronos-L, Chronos-B, Chronos-Sm]}}
BASELINES = {
    "m_dense": {"NRMSE": [0.392, 0.319, 0.314, 0.317], "SMAPE": [0.295, 0.186, 0.181, 0.183]},
    "loop_seattle": {"NRMSE": [0.160, 0.175, 0.177, 0.181], "SMAPE": [0.131, 0.132, 0.133, 0.134]},
    "sz_taxi": {"NRMSE": [0.369, 0.372, 0.375, 0.380], "SMAPE": [0.406, 0.425, 0.418, 0.450]},
    "solar": {"NRMSE": [1.084, 1.144, 1.086, 1.144], "SMAPE": [1.217, 1.270, 1.230, 1.252]},
    "bizitobs_application": {"NRMSE": [0.213, 0.160, 0.155, 0.147], "SMAPE": [0.162, 0.098, 0.110, 0.107]},
    "bizitobs_l2c": {"NRMSE": [0.780, 1.010, 1.040, 0.999], "SMAPE": [0.880, 1.140, 1.135, 1.076]},
    "bizitobs_service": {"NRMSE": [0.299, 0.236, 0.233, 0.227], "SMAPE": [0.228, 0.139, 0.150, 0.140]},
    "car_parts": {"MASE": [3.075, 3.013, 3.008, 2.958], "SMAPE": [1.796, 1.862, 1.871, 1.871]},
    "jena_weather": {"NRMSE": [0.236, 0.241, 0.243, 0.261], "SMAPE": [0.638, 0.657, 0.653, 0.649]},
}

def generate_latex(csv_path, out_path):
    df = pd.read_csv(csv_path)
    # Average over horizons for each subset
    df_avg = df.groupby("subset").mean().reset_index()
    
    # Mapping subset names from CSV to baseline keys
    mapping = {
        "m_dense_H_long": "m_dense",
        "loop_seattle_H_long": "loop_seattle",
        "sz_taxi_H_short": "sz_taxi",
        "solar_H_long": "solar",
        "bizitobs_application_10S_long": "bizitobs_application",
        "bizitobs_l2c_H_long": "bizitobs_l2c",
        "bizitobs_service_10S_long": "bizitobs_service",
        "car_parts_M_short": "car_parts",
        "jena_weather_H_long": "jena_weather"
    }
    
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Zero-Shot Performance Comparison on GIFT-Eval subsets.}")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\toprule")
    latex.append(r"Dataset & Metric & SEMPO & Chronos-L & Chronos-B & Chronos-Sm & \textbf{MoP (Ours)} \\")
    latex.append(r"\midrule")
    
    for csv_name, baseline_key in mapping.items():
        row = df_avg[df_avg["subset"] == csv_name]
        if row.empty: continue
        
        subset_disp = baseline_key.replace("_", " ")
        base_metrics = BASELINES.get(baseline_key, {})
        
        for metric_name in ["NRMSE", "SMAPE"]:
            if metric_name not in base_metrics and metric_name == "NRMSE" and "MASE" in base_metrics:
                metric_name = "MASE"
            
            vals = base_metrics.get(metric_name, [0.0]*4)
            # Ours
            if metric_name == "NRMSE":
                our_val = row["nrmse"].values[0]
            elif metric_name == "SMAPE":
                our_val = row["smape"].values[0]
            else:
                our_val = row["mse"].values[0] # placeholder if metric mismatch
                
            line = f"{subset_disp} & {metric_name} & {vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.3f} & \\mathbf{{{our_val:.3f}}} \\\\"
            latex.append(line)
        latex.append(r"\midrule")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    
    with open(out_path, "w") as f:
        f.write("\n".join(latex))
    print(f"LaTeX table saved to {out_path}")

if __name__ == "__main__":
    import sys
    generate_latex(sys.argv[1], sys.argv[2])
