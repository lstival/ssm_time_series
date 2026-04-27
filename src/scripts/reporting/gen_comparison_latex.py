import csv
import os
from pathlib import Path

def get_best_and_second(v1, v2, v3):
    vals = [(v, i) for i, v in enumerate([v1, v2, v3]) if v is not None]
    if not vals: return [None]*3
    sorted_vals = sorted(vals)
    best_idx = sorted_vals[0][1]
    second_idx = sorted_vals[1][1] if len(sorted_vals) > 1 else None
    
    res = [f"{v:.4f}" if v is not None else "N/A" for v in [v1, v2, v3]]
    res[best_idx] = r"\textcolor{red}{\textbf{" + res[best_idx] + "}}"
    if second_idx is not None:
        res[second_idx] = r"\textcolor{blue}{\underline{" + res[second_idx] + "}}"
    return res

def parse_csv(path):
    data = {}
    if not os.path.exists(path): return data
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row['dataset']
            h = int(row['horizon'])
            mse = float(row['mse'])
            mae = float(row['mae'])
            data[(ds, h)] = (mse, mae)
    return data

def main():
    root = Path("/home/WUR/stiva001/WUR/ssm_time_series")
    results_dir = root / "results"
    
    # Paths for Multimodal (MoMS)
    moms_zs = parse_csv(results_dir / "moms_full_pipeline" / "mop_zeroshot_clip_results.csv")
    moms_fs = parse_csv(results_dir / "moms_full_pipeline" / "mop_fewshot_clip_film_results.csv")
    
    # Paths for Unimodal Temporal (GIFT Boosted)
    uni_zs = parse_csv(results_dir / "moms_clip_uni_gift_ssl" / "mop_zeroshot_clip_uni_results.csv")
    uni_fs = parse_csv(results_dir / "moms_clip_uni_gift_ssl" / "mop_fewshot_clip_uni_results.csv")
    
    # Visual Baseline (Unimodal Visual)
    vis_zs = parse_csv(results_dir / "simclr_visual_nano" / "probe_lotsa_results.csv") # Approximated as ZS baseline
    
    datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "weather", "traffic", "electricity"]
    horizons = [96, 192, 336, 720]
    
    tex = r"""\begin{table}[t!]
\centering
\caption{Zero-Shot (ZS) and Few-Shot (FS) evaluation comparing Unimodal and Multimodal MoMS. \textcolor{red}{\textbf{Red}}: Best, \textcolor{blue}{\underline{Blue}}: 2nd best.}
\label{tab:automated_comparison}
\begin{tabular}{ll ccc ccc}
\toprule
& & \multicolumn{3}{c}{\textbf{Zero-Shot (MSE)}} & \multicolumn{3}{c}{\textbf{Few-Shot (MSE)}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
\textbf{Dataset} & \textbf{H} & \textbf{Temporal} & \textbf{Visual} & \textbf{Multimodal} & \textbf{Temporal} & \textbf{Visual} & \textbf{Multimodal} \\
\midrule
"""
    for ds in datasets:
        ds_name = ds.capitalize() if ds in ["weather", "traffic", "electricity"] else ds
        for i, h in enumerate(horizons):
            # Zero Shot
            v_t_zs = uni_zs.get((ds, h), (None, None))[0]
            v_v_zs = vis_zs.get((ds, h), (None, None))[0]
            v_m_zs = moms_zs.get((ds, h), (None, None))[0]
            zs_row = get_best_and_second(v_t_zs, v_v_zs, v_m_zs)
            
            # Few Shot
            v_t_fs = uni_fs.get((ds, h), (None, None))[0]
            v_v_fs = None # No 5% FS visual-only specialized run usually, using visual baseline
            v_m_fs = moms_fs.get((ds, h), (None, None))[0]
            fs_row = get_best_and_second(v_t_fs, v_v_fs, v_m_fs)
            
            row = f" {ds_name if i==0 else ''} & {h} & {' & '.join(zs_row)} & {' & '.join(fs_row)} \\\\\n"
            tex += row
        tex += r"\midrule" + "\n"
        
    tex += r"\bottomrule\end{tabular}\end{table}"
    
    with open(results_dir / "full_comparison_table.tex", "w") as f:
        f.write(tex)
    print(f"Table saved to {results_dir / 'full_comparison_table.tex'}")

if __name__ == "__main__":
    main()
