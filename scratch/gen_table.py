import csv
import os

datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'weather', 'traffic', 'electricity']
horizons = [96, 192, 336, 720]
methods = ['clip', 'gram', 'byol', 'simclr']

results = {}

for method in methods:
    file_path = f"results/moms_full_pipeline/mop_fewshot_{method}_film_results.csv"
    if not os.path.exists(file_path):
        # try without film
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

# Sort and format
table_rows = []
for ds in datasets:
    ds_label = ds
    if ds == 'weather': ds_label = 'Weather'
    if ds == 'traffic': ds_label = 'Traffic'
    if ds == 'electricity': ds_label = 'Electricity'
    
    for i, h in enumerate(horizons):
        row_data = results.get((ds, h), {})
        vals = []
        for method in methods:
            vals.append(row_data.get(method, 0.0))
        
        # Determine best and second best
        sorted_vals = sorted([(v, m) for m, v in row_data.items() if v != 0])
        best_val = sorted_vals[0][0] if len(sorted_vals) > 0 else 0
        second_best_val = sorted_vals[1][0] if len(sorted_vals) > 1 else 0
        
        row_str = f" & {h}"
        for method in methods:
            val = row_data.get(method, 0.0)
            val_str = f"{val:.4f}"
            if abs(val - best_val) < 1e-7 and val != 0:
                row_str += " & {\\textcolor{red}{\\textbf{" + val_str + "}}}"
            elif abs(val - second_best_val) < 1e-7 and val != 0:
                row_str += " & {\\textcolor{blue}{\\underline{" + val_str + "}}}"
            else:
                row_str += f" & {val_str}"
        
        if i == 0:
            prefix = f"\\multirow{{4}}{{*}}{{{ds_label}}} \n"
        else:
            prefix = ""
        table_rows.append(f"{prefix}{row_str} \\\\")
    table_rows.append("\\midrule")

# Remove last midrule and add bottomrule
if table_rows:
    table_rows[-1] = "\\bottomrule"

latex_table = """
\\begin{table}[!ht]
\\centering
\\caption{Linear probe MSE results for MoMS-aligned nano methods across 7 datasets and 4 forecasting horizons. \\textcolor{red}{\\textbf{Red bold}} = best, \\textcolor{blue}{\\underline{blue underline}} = second best (per dataset/horizon).}
\\label{tab:moms_aligned_mse}
\\renewcommand{\\arraystretch}{0.95} 
\\resizebox{\\linewidth}{!}{%
\\begin{tabular}{llrrrr}
\\toprule
\\textbf{Dataset} & \\textbf{H} & \\textbf{CLIP-MoMS} & \\textbf{GRAM-MoMS} & \\textbf{BYOL-MoMS} & \\textbf{SimCLR-MoMS} \\\\
\\midrule
""" + "\n".join(table_rows) + """
\\end{tabular}%
}
\\end{table}
"""

print(latex_table)
