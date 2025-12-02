import pandas as pd
import re
import os

# -----------------------

file_name = "icml_zeroshot_forecast_20251120_1310.csv"
file_path = f"../results/{file_name}"
df = pd.read_csv(file_path)



# -----------------------
def clean_name(path):
    filename = path.replace("\\", "/").split("/")[-1]
    name = re.sub(r"\.(csv|txt|npz)$", "", filename)

    # Standardize
    if "behind_electricity" in name:
        return "Electricity (behind)"
    if "middle_electricity" in name:
        return "Electricity (middle)"
    if name == "electricity":
        return "Electricity"
    if name == "exchange_rate":
        return "Exchange"
    if name == "solar_AL":
        return "Solar"
    if name == "PEMS04":
        return "Traffic"

    return name

df["dataset"] = df["dataset_name"].apply(clean_name)

# -----------------------

df = df[["dataset", "horizon", "mae", "mse"]]

# Horizons sorted
HORIZONS = [96, 192, 336, 720]

# -----------------------
table_data = {}
datasets = df["dataset"].unique()

for dataset in datasets:
    entry = []
    for h in HORIZONS:
        subset = df[(df["dataset"] == dataset) & (df["horizon"] == h)]
        if subset.empty:
            entry.extend(["-", "-"])
        else:
            mae = subset["mae"].iloc[0]
            mse = subset["mse"].iloc[0]
            entry.extend([f"{mae:.5f}", f"{mse:.5f}"])
    table_data[dataset] = entry

# -----------------------
latex = []

latex.append(r"\begin{table*}[ht!]")
latex.append(r"\centering")
latex.append(r"\caption{MAE and MSE metrics across datasets and horizons.}")
latex.append(r"\small")
latex.append(r"\begin{adjustbox}{max width=\textwidth}")
latex.append(
    r"\begin{tabular}{"
    r"l"
    r"*{4}{>{\centering\arraybackslash}p{1.4cm}"
    r">{\centering\arraybackslash}p{1.4cm}}"
    r"}"
)
latex.append(r"\toprule")

# Header row
latex.append(
    r"\multirow{2}{*}{\textbf{Dataset}} "
    r"& \multicolumn{2}{c}{\textbf{96}} "
    r"& \multicolumn{2}{c}{\textbf{192}} "
    r"& \multicolumn{2}{c}{\textbf{336}} "
    r"& \multicolumn{2}{c}{\textbf{720}} \\"
)

latex.append(r"\cmidrule(lr){2-3}")
latex.append(r"\cmidrule(lr){4-5}")
latex.append(r"\cmidrule(lr){6-7}")
latex.append(r"\cmidrule(lr){8-9}")

latex.append(
    r"& \textbf{MAE} & \textbf{MSE}"
    r"& \textbf{MAE} & \textbf{MSE}"
    r"& \textbf{MAE} & \textbf{MSE}"
    r"& \textbf{MAE} & \textbf{MSE} \\"
)

latex.append(r"\midrule")
latex.append("")

# Body rows
for dataset in sorted(table_data.keys()):
    row = " & ".join([dataset] + table_data[dataset]) + r" \\"
    latex.append(row)

latex.append(r"\bottomrule")
latex.append(r"\end{tabular}")
latex.append(r"\end{adjustbox}")
latex.append(r"\end{table*}")

# Print final LaTeX
final_latex = "\n".join(latex)
# print(final_latex)

out_filename = f"results_all_ICML_datasets_{file_name.replace('.csv', '')}.tex"
out_dir = os.path.dirname(file_name) or os.getcwd()
out_path = os.path.join(out_dir, out_filename)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(final_latex)

print(f"Wrote LaTeX to: {out_path}")