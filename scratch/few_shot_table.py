import re

raw_data = """
ETTh1 0.406 0.423 0.382 0.405 0.627 0.543 0.681 0.560 0.642 0.546 1.070 0.710 0.750 0.611 0.694 0.569 0.925 0.647 0.943 0.646 0.658 0.562 0.722 0.598
ETTh2 0.320 0.372 0.333 0.376 0.382 0.418 0.400 0.433 0.380 0.415 0.488 0.475 0.694 0.577 0.827 0.615 0.439 0.448 0.470 0.489 0.463 0.454 0.441 0.457
ETTm1 0.363 0.385 0.389 0.389 0.425 0.434 0.472 0.450 0.416 0.421 0.784 0.597 0.400 0.417 0.526 0.476 0.717 0.561 0.857 0.598 0.730 0.592 0.796 0.620
ETTm2 0.256 0.315 0.285 0.328 0.274 0.323 0.308 0.346 0.279 0.325 0.356 0.388 0.399 0.426 0.314 0.352 0.344 0.372 0.341 0.372 0.381 0.404 0.388 0.433
Weather 0.230 0.268 0.236 0.273 0.260 0.309 0.263 0.301 0.257 0.295 0.309 0.339 0.263 0.308 0.269 0.303 0.298 0.318 0.327 0.328 0.309 0.353 0.310 0.353
Electricity 0.165 0.263 0.183 0.276 0.179 0.268 0.178 0.273 0.186 0.281 0.201 0.296 0.176 0.275 0.181 0.277 0.402 0.453 0.627 0.603 0.266 0.353 0.346 0.404
Traffic 0.410 0.287 0.427 0.303 0.423 0.298 0.434 0.305 0.419 0.298 0.450 0.324 0.450 0.317 0.418 0.296 0.867 0.493 1.526 0.839 0.676 0.423 0.833 0.502
"""

models = [
    "CM-Mamba", "SEMPO", "TTM", "Time-LLM", "GPT4TS", "S2IP-LLM", "iTransformer", 
    "DLinear", "PatchTST", "TimesNet", "Stationary", "FEDformer", "Autoformer"
]

our_vals = {
    'ETTh1': ['0.333', '0.492'],
    'ETTh2': ['0.365', '0.489'],
    'ETTm1': ['0.361', '0.515'],
    'ETTm2': ['0.254', '0.404'],
    'Weather': ['0.972', '0.732'],
    'Electricity': ['0.789', '0.675'],
    'Traffic': ['0.582', '0.578'],
}

lines = raw_data.strip().split('\n')
parsed = {}
for line in lines:
    parts = line.split()
    ds = parts[0]
    vals = parts[1:]
    # Prepend CM-Mamba
    vals = our_vals[ds] + vals
    parsed[ds] = vals

# Calculate red/blue
out_lines = []
red_counts = {m: 0 for m in models}

header1 = r"\textbf{Models} & \multicolumn{2}{c}{\textbf{CM-Mamba}} & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in models[1:]]) + r" \\"
cmids = " ".join([f"\\cmidrule(lr){{{(i*2)+2}-{(i*2)+3}}}" for i in range(len(models))])
metrics = r"\textbf{Metrics} & " + " & ".join(["MSE & MAE" for _ in models]) + r" \\"

for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']:
    vals = parsed[ds]
    mses = [(i, float(vals[i*2])) for i in range(len(models))]
    maes = [(i, float(vals[i*2+1])) for i in range(len(models))]
    
    mses.sort(key=lambda x: x[1])
    maes.sort(key=lambda x: x[1])
    
    red_mse_idx = mses[0][0]
    blue_mse_idx = mses[1][0]
    
    red_mae_idx = maes[0][0]
    blue_mae_idx = maes[1][0]
    
    row_str = f"{ds}"
    for i in range(len(models)):
        mse_val = vals[i*2]
        if i == red_mse_idx:
            mse_fmt = rf"\textcolor{{red}}{{\textbf{{{mse_val}}}}}"
            red_counts[models[i]] += 1
        elif i == blue_mse_idx:
            mse_fmt = rf"\textcolor{{blue}}{{\underline{{{mse_val}}}}}"
        else:
            mse_fmt = mse_val
            
        mae_val = vals[i*2+1]
        if i == red_mae_idx:
            mae_fmt = rf"\textcolor{{red}}{{\textbf{{{mae_val}}}}}"
            red_counts[models[i]] += 1
        elif i == blue_mae_idx:
            mae_fmt = rf"\textcolor{{blue}}{{\underline{{{mae_val}}}}}"
        else:
            mae_fmt = mae_val
            
        row_str += f" & {mse_fmt} & {mae_fmt}"
    row_str += r" \\"
    out_lines.append(row_str)

# count row
cnt_str = r"\textit{1st Count}"
for i, m in enumerate(models):
    cnt_str += r" & \multicolumn{2}{c}{" + (r"\textbf{" + str(red_counts[m]) + "}" if red_counts[m] > 0 else "0") + "}"
cnt_str += r" \\"

full_tex = r"""\begin{table*}[t]
\centering
\caption{Few-shot results on the TSLib benchmark with 5\% training data. MSE and MAE are averaged over forecasting horizons $H \in \{96, 192, 336, 720\}$, where lower values indicate better prediction. \textcolor{red}{\textbf{Red}}: the best, \textcolor{blue}{\underline{Blue}}: the second best.}
\label{tab:few_shot_5pct}
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{l""" + "c"*(len(models)*2) + r"""}
\toprule
""" + header1 + "\n" + cmids + "\n" + metrics + r"""
\midrule
""" + "\n".join(out_lines) + "\n" + r"""\midrule
""" + cnt_str + "\n" + r"""\bottomrule
\end{tabular}
}
\end{table*}
"""

print(full_tex)
