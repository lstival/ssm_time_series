import os
import csv
from pathlib import Path

# --- Configurações de Caminhos e Métodos ---
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "ssl_detailed_results.tex"

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "traffic", "electricity", "exchange_rate", "solar_AL"]
HORIZONS = [96, 192, 336, 720]

TIERS = {
    "micro": {
        "SimCLR": "icml_simclr_micro",
        "CLIP": "icml_clip_micro",
        "BYOL": "icml_byol_micro",
        "VL-JEPA": "icml_vl_jepa_micro",
        "GRAM": "icml_gram_micro"
    },
    "nano": {
        "SimCLR-Bi": "probe_simclr_bimodal_nano",
        "CLIP": "probe_clip_nano",
        "BYOL-Bi": "probe_byol_bimodal_nano",
        "VL-JEPA": "probe_vl_jepa_nano",
        "GRAM": "probe_gram_nano"
    }
}

def get_csv_path(method_dir, dataset):
    if dataset == "exchange_rate":
        psn_dir = RESULTS_DIR / f"{method_dir}_psn"
        if psn_dir.exists():
            method_dir = psn_dir.name
            
    base_path = RESULTS_DIR / method_dir
    for fname in ["probe_results_full.csv", "probe_lotsa_results.csv"]:
        path = base_path / fname
        if path.exists():
            return path
    return None

def load_all_results():
    data = {}
    for tier, methods in TIERS.items():
        data[tier] = {}
        for method_label, method_dir in methods.items():
            combined_rows = []
            for ds in DATASETS:
                path = get_csv_path(method_dir, ds)
                if path:
                    with open(path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row['dataset'] = row['dataset'].replace(".csv", "").lower()
                            combined_rows.append(row)
            data[tier][method_label] = combined_rows
    return data

def format_value(val, is_best, is_second):
    if val is None or val >= 999999: return "--"
    s = f"{val:.3f}"
    if is_best:
        return f"\\textcolor{{red}}{{\\textbf{{{s}}}}}"
    if is_second:
        return f"\\underline{{\\textcolor{{blue}}{{{s}}}}}"
    return s

def generate_latex_table(tier_name, tier_data):
    methods = list(TIERS[tier_name].keys())
    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Detailed SSL {tier_name.capitalize()} tier performance per horizon.}}",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{ll" + "cc" * len(methods) + "}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{H} & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{{m}}}}}" for m in methods]) + " \\\\",
        "\\cmidrule(lr){3-4}" + "".join([f"\\cmidrule(lr){{{(i*2)+5}-{(i*2)+6}}}" for i in range(len(methods)-1)]) + " \\\\",
        " & & " + " & ".join(["MSE & MAE" for _ in methods]) + " \\\\",
        "\\midrule"
    ]

    for ds in DATASETS:
        ds_label = ds.replace("_", "\\_").capitalize()
        for h_idx, h in enumerate(HORIZONS):
            row = [ds_label if h_idx == 0 else "", str(h)]
            mses, maes = [], []
            for m in methods:
                rows = tier_data.get(m, [])
                match = [r for r in rows if r['dataset'] == ds.lower() and int(r['horizon']) == h]
                if match:
                    mses.append(float(match[0]['mse']))
                    maes.append(float(match[0]['mae']))
                else:
                    mses.append(999999.0); maes.append(999999.0)

            valid_mses = sorted([v for v in mses if v < 999999])
            valid_maes = sorted([v for v in maes if v < 999999])
            
            for m_idx, (mse, mae) in enumerate(zip(mses, maes)):
                is_best_mse = (mse == valid_mses[0]) if valid_mses else False
                is_sec_mse = (len(valid_mses) > 1 and mse == valid_mses[1])
                is_best_mae = (mae == valid_maes[0]) if valid_maes else False
                is_sec_mae = (len(valid_maes) > 1 and mae == valid_maes[1])
                row.append(format_value(mse, is_best_mse, is_sec_mse))
                row.append(format_value(mae, is_best_mae, is_sec_mae))
            latex.append(" & ".join(row) + " \\\\")
        latex.append("\\midrule")
    latex.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""])
    return "\n".join(latex)

def main():
    all_data = load_all_results()
    with open(OUTPUT_FILE, "w") as f:
        f.write("% Generated SSL Comparison Tables\n\n")
        f.write("\\usepackage{booktabs}\n\\usepackage{xcolor}\n\\usepackage{graphicx}\n\n")
        for tier in ["micro", "nano"]:
            f.write(generate_latex_table(tier, all_data[tier]))
            f.write("\n\n")
    print(f"Success! Tables saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
