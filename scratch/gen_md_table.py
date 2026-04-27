import csv
import os

datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'weather', 'traffic', 'electricity']
horizons = [96, 192, 336, 720]
methods = ['clip', 'gram', 'byol', 'simclr']

results = {}

for method in methods:
    file_path = f"results/moms_full_pipeline/mop_fewshot_{method}_film_results.csv"
    if not os.path.exists(file_path):
        file_path = f"results/moms_full_pipeline/mop_fewshot_{method}_results.csv"
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row['dataset']
                h = int(row['horizon'])
                mse = float(row['mse'])
                if ds in datasets and h in horizons:
                    if (ds, h) not in results:
                        results[(ds, h)] = {}
                    results[(ds, h)][method] = mse

# Header
md_table = "| Dataset | H | CLIP-MoMS | GRAM-MoMS | BYOL-MoMS | SimCLR-MoMS |\n"
md_table += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"

for ds in datasets:
    ds_label = ds.capitalize() if ds in ['weather', 'traffic', 'electricity'] else ds
    for i, h in enumerate(horizons):
        row_data = results.get((ds, h), {})
        
        # Rankings
        sorted_vals = sorted([(v, m) for m, v in row_data.items() if v != 0])
        best_val = sorted_vals[0][0] if len(sorted_vals) > 0 else 0
        second_val = sorted_vals[1][0] if len(sorted_vals) > 1 else 0
        
        row_str = f"| {ds_label if i == 0 else ''} | {h} |"
        for method in methods:
            val = row_data.get(method, 0.0)
            val_str = f"{val:.4f}"
            if abs(val - best_val) < 1e-7 and val != 0:
                row_str += f" **{val_str}** |"
            elif abs(val - second_val) < 1e-7 and val != 0:
                row_str += f" _{val_str}_ |"
            else:
                row_str += f" {val_str} |"
        md_table += row_str + "\n"
    md_table += "| --- | --- | --- | --- | --- | --- |\n"

print(md_table)
