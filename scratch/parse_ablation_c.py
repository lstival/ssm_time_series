import re
from collections import defaultdict

log_file = "/home/WUR/stiva001/WUR/ssm_time_series/logs/ablation_C/train_66417573.out"

with open(log_file, "r") as f:
    content = f.read()

# Variants are separated by "==========================================================\nVariant: "
parts = content.split("============================================================\nVariant: ")
data = {}

for part in parts[1:]:
    lines = part.split("\n")
    variant_name = lines[0].strip()
    if variant_name == "concat_supervised":
        continue # handled separately if needed
    
    variant_data = defaultdict(lambda: {"mse": [], "mae": []})
    
    for line in lines:
        # Match pattern:   ETTm1.csv  H= 96  MSE=0.2486  MAE=0.4215
        match = re.search(r"(\S+\.csv)\s+H=\s*(\d+)\s+MSE=([\d.]+)\s+MAE=([\d.]+)", line)
        if match:
            ds = match.group(1).replace(".csv", "")
            mse = float(match.group(3))
            mae = float(match.group(4))
            variant_data[ds]["mse"].append(mse)
            variant_data[ds]["mae"].append(mae)
    
    data[variant_name] = variant_data

variants = ["clip_symm", "cosine_mse", "unimodal_temporal"]
datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "weather", "traffic", "electricity", "exchange_rate"]

print("Variant,Dataset,Avg MSE,Avg MAE")
for v in variants:
    v_dict = data.get(v, {})
    for ds in datasets:
        metrics = v_dict.get(ds, {})
        if metrics["mse"]:
            avg_mse = sum(metrics["mse"]) / len(metrics["mse"])
            avg_mae = sum(metrics["mae"]) / len(metrics["mae"])
            print(f"{v},{ds},{avg_mse:.4f},{avg_mae:.4f}")
        else:
            print(f"{v},{ds},nan,nan")
