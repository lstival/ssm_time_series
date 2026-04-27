import re

tex = r"""
\begin{table*}[t]
\centering
\caption{Forecasting comparison. Our CM-Mamba uses a \emph{linear probe} on frozen representations; all other models are zero-shot foundation models. Avg MSE/MAE over $H\in\{96,192,336,720\}$. \textcolor{red}{\textbf{Red}}: best, \textcolor{blue}{\underline{Blue}}: second-best.}
\renewcommand{\arraystretch}{1.2}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccccccccccccccccccccc}
\toprule
\textbf{Dataset} & \multicolumn{2}{c}{\textbf{CM-Mamba-nano (ours)}} & \multicolumn{2}{c}{\textit{SEMPO}} & \multicolumn{2}{c}{\textit{Time-MoE-B}} & \multicolumn{2}{c}{\textit{Time-MoE-L}} & \multicolumn{2}{c}{\textit{Timer}} & \multicolumn{2}{c}{\textit{Moirai-S}} & \multicolumn{2}{c}{\textit{Moirai-B}} & \multicolumn{2}{c}{\textit{Moirai-L}} & \multicolumn{2}{c}{\textit{Chronos-S}} & \multicolumn{2}{c}{\textit{Chronos-B}} & \multicolumn{2}{c}{\textit{Chronos-L}} & \multicolumn{2}{c}{\textit{TimesFM}} & \multicolumn{2}{c}{\textit{Moment}} \\
 & \multicolumn{2}{c}{\scriptsize{4.5M}} & \multicolumn{2}{c}{\scriptsize{6.5M}} & \multicolumn{2}{c}{\scriptsize{113M}} & \multicolumn{2}{c}{\scriptsize{453M}} & \multicolumn{2}{c}{\scriptsize{67.4M}} & \multicolumn{2}{c}{\scriptsize{14M}} & \multicolumn{2}{c}{\scriptsize{91M}} & \multicolumn{2}{c}{\scriptsize{311M}} & \multicolumn{2}{c}{\scriptsize{46M}} & \multicolumn{2}{c}{\scriptsize{200M}} & \multicolumn{2}{c}{\scriptsize{710M}} & \multicolumn{2}{c}{\scriptsize{200M}} & \multicolumn{2}{c}{\scriptsize{385M}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13} \cmidrule(lr){14-15} \cmidrule(lr){16-17} \cmidrule(lr){18-19} \cmidrule(lr){20-21} \cmidrule(lr){22-23} \cmidrule(lr){24-25} \cmidrule(lr){26-27}
 & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\
\midrule
ETTh1 & \textcolor{red}{\textbf{0.342}} & 0.502 & \textcolor{blue}{\underline{0.410}} & \textcolor{red}{\textbf{0.430}} & 0.445 & 0.449 & 0.435 & 0.449 & 0.451 & 0.463 & 0.448 & 0.432 & 0.433 & \textcolor{blue}{\underline{0.431}} & 0.466 & 0.443 & 0.551 & 0.463 & 0.524 & 0.439 & 0.541 & 0.443 & 0.489 & 0.444 & 0.708 & 0.580 \\
ETTh2 & \textcolor{red}{\textbf{0.243}} & \textcolor{blue}{\underline{0.393}} & \textcolor{blue}{\underline{0.341}} & \textcolor{red}{\textbf{0.391}} & 0.566 & 0.479 & 0.477 & 0.452 & 0.366 & 0.408 & 0.355 & 0.401 & 0.360 & 0.399 & 0.382 & 0.397 & 0.394 & 0.409 & 0.392 & 0.401 & 0.385 & 0.400 & 0.396 & 0.405 & 0.392 & 0.430 \\
ETTm1 & \textcolor{red}{\textbf{0.393}} & 0.546 & 0.503 & 0.466 & 0.507 & 0.480 & 0.483 & 0.471 & 0.544 & 0.476 & 0.554 & 0.477 & 0.566 & 0.464 & 0.601 & 0.468 & 0.628 & 0.487 & 0.566 & 0.465 & 0.521 & \textcolor{blue}{\underline{0.448}} & \textcolor{blue}{\underline{0.434}} & \textcolor{red}{\textbf{0.419}} & 0.697 & 0.555 \\
ETTm2 & \textcolor{red}{\textbf{0.273}} & 0.410 & \textcolor{blue}{\underline{0.286}} & \textcolor{red}{\textbf{0.341}} & 0.538 & 0.463 & 0.509 & 0.452 & 0.298 & 0.346 & 0.323 & 0.351 & 0.339 & 0.356 & 0.334 & 0.352 & 0.320 & 0.355 & 0.308 & \textcolor{blue}{\underline{0.344}} & 0.315 & 0.350 & 0.320 & 0.353 & 0.319 & 0.360 \\
Weather & $-^\dagger$ & $-^\dagger$ & \textcolor{red}{\textbf{0.248}} & \textcolor{red}{\textbf{0.287}} & 0.279 & 0.309 & 0.318 & 0.334 & 0.292 & 0.312 & \textcolor{blue}{\underline{0.267}} & 0.306 & 0.312 & 0.295 & 0.477 & \textcolor{blue}{\underline{0.289}} & 0.298 & 0.302 & 0.283 & 0.295 & 0.292 & 0.297 & -- & -- & 0.291 & 0.323 \\
Electricity & 0.453 & 0.499 & \textcolor{red}{\textbf{0.196}} & \textcolor{red}{\textbf{0.295}} & -- & -- & -- & -- & 0.297 & 0.375 & 0.243 & 0.329 & \textcolor{blue}{\underline{0.207}} & \textcolor{blue}{\underline{0.296}} & 0.224 & 0.309 & 0.246 & 0.312 & 0.336 & 0.329 & 0.326 & 0.328 & -- & -- & 0.861 & 0.766 \\
Traffic & \textcolor{red}{\textbf{0.324}} & \textcolor{blue}{\underline{0.406}} & \textcolor{blue}{\underline{0.466}} & \textcolor{red}{\textbf{0.344}} & -- & -- & -- & -- & 0.613 & 0.407 & -- & -- & -- & -- & -- & -- & 0.614 & 0.420 & 0.603 & 0.413 & 0.600 & 0.411 & -- & -- & 1.411 & 0.804 \\
\midrule
\textit{1\textsuperscript{st} count} & \multicolumn{2}{c}{\textbf{5}} & \multicolumn{2}{c}{\textbf{2}} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} & \multicolumn{2}{c}{0} \\
\bottomrule
\end{tabular}
}
\label{tab:zero-shot}
\end{table*}
"""

new_vals = {
    'ETTh1': ['0.333', '0.492'],
    'ETTh2': ['0.365', '0.489'],
    'ETTm1': ['0.361', '0.515'],
    'ETTm2': ['0.254', '0.404'],
    'Weather': ['0.972', '0.732'],
    'Electricity': ['0.789', '0.675'],
    'Traffic': ['0.582', '0.578'],
}

def clean_val(v):
    if '--' in v or '-' in v: return float('inf')
    v = v.replace(r'\textcolor{red}{\textbf{', '').replace('}}', '')
    v = v.replace(r'\textcolor{blue}{\underline{', '').replace('}}', '')
    try:
        return float(v)
    except:
        return float('inf')

lines = tex.split('\n')
out_lines = []
red_counts = [0] * 13 # 1 for CM, 12 for others

for idx, line in enumerate(lines):
    if line.startswith(('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')):
        parts = [p.strip() for p in line.split('&')]
        ds = parts[0].strip()
        num_models = (len(parts) - 1) // 2
        
        # update first model
        parts[1] = new_vals[ds][0]
        parts[2] = new_vals[ds][1]

        # Extract values
        cols_mse = []
        cols_mae = []
        for i in range(num_models):
            cols_mse.append((i, clean_val(parts[1 + i*2])))
            cols_mae.append((i, clean_val(parts[2 + i*2])))
            
        def rank(lst):
            return sorted(lst, key=lambda x: x[1])
            
        ranked_mse = rank(cols_mse)
        ranked_mae = rank(cols_mae)
        
        red_mse_idx = ranked_mse[0][0]
        blue_mse_idx = ranked_mse[1][0] if len(ranked_mse) > 1 else None
        
        red_mae_idx = ranked_mae[0][0]
        blue_mae_idx = ranked_mae[1][0] if len(ranked_mae) > 1 else None
        
        def format_val(i, v, red_idx, blue_idx):
            if clean_val(v) == float('inf'): return "--"
            if i == red_idx:
                red_counts[i] += 1
                return r'\textcolor{red}{\textbf{' + v.replace(r'\textcolor{red}{\textbf{','').replace('}}','').replace(r'\textcolor{blue}{\underline{','') + '}}'
            elif i == blue_idx:
                return r'\textcolor{blue}{\underline{' + v.replace(r'\textcolor{blue}{\underline{','').replace('}}','').replace(r'\textcolor{red}{\textbf{','') + '}}'
            else:
                return v.replace(r'\textcolor{red}{\textbf{','').replace('}}','').replace(r'\textcolor{blue}{\underline{','')
                
        for i in range(num_models):
            parts[1 + i*2] = format_val(i, parts[1 + i*2], red_mse_idx, blue_mse_idx)
            parts[2 + i*2] = format_val(i, parts[2 + i*2], red_mae_idx, blue_mae_idx)
            
        out_lines.append(' & '.join(parts) + ' \\\\')
    elif line.startswith(r'\textit{1\textsuperscript{st} count}'):
        parts = [p.strip() for p in line.split('&')]
        for i in range(13):
            # i+1 is the position in parts
            c = str(red_counts[i])
            if i == 0 or i == 1: # We usually bold the top 2
                parts[1+i] = r'\multicolumn{2}{c}{\textbf{' + c + '}}'
            else:
                parts[1+i] = r'\multicolumn{2}{c}{' + c + '}'
        # Just bold the actual best and second best counts
        out_lines.append(' & '.join(parts) + ' \\\\')
    else:
        out_lines.append(line)

print('\n'.join(out_lines))
