import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def load_and_process_data(csv_path):
    """Load and process the CSV data"""
    df = pd.read_csv(csv_path)
    
    # Create a mapping from dataset names in CSV to standard names
    dataset_mapping = {
        'ETTh1.csv': 'ETTh1',
        'ETTh2.csv': 'ETTh2',
        'ETTm1.csv': 'ETTm1',
        'ETTm2.csv': 'ETTm2',
        'electricity.csv': 'Electricity',
        'behind_electricity.csv': 'Electricity',
        'middle_electricity.csv': 'Electricity',
        'exchange_rate.csv': 'Exchange',
        'traffic.csv': 'Traffic',
        'weather.csv': 'Weather'
    }
    
    # Map dataset names
    df['standard_dataset'] = df['dataset_name'].map(dataset_mapping)
    
    # For electricity datasets, we'll use the main electricity.csv results
    df_filtered = df[~df['dataset_name'].isin(['behind_electricity.csv', 'middle_electricity.csv'])].copy()
    
    return df_filtered

def create_metrics_plot(df):
    """Create visualization plots for the metrics"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Across Datasets and Horizons', fontsize=16, fontweight='bold')
    
    # Plot 1: MSE by dataset and horizon
    pivot_mse = df.pivot_table(values='mse', index='standard_dataset', columns='horizon', aggfunc='mean')
    sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('Mean Squared Error (MSE) by Dataset and Horizon')
    axes[0,0].set_xlabel('Forecast Horizon')
    axes[0,0].set_ylabel('Dataset')
    
    # Plot 2: MAE by dataset and horizon
    pivot_mae = df.pivot_table(values='mae', index='standard_dataset', columns='horizon', aggfunc='mean')
    sns.heatmap(pivot_mae, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('Mean Absolute Error (MAE) by Dataset and Horizon')
    axes[0,1].set_xlabel('Forecast Horizon')
    axes[0,1].set_ylabel('Dataset')
    
    # Plot 3: Average MSE across all horizons by dataset
    avg_mse = df.groupby('standard_dataset')['mse'].mean().sort_values()
    bars1 = axes[1,0].bar(range(len(avg_mse)), avg_mse.values, color='skyblue', alpha=0.7)
    axes[1,0].set_title('Average MSE Across All Horizons')
    axes[1,0].set_xlabel('Dataset')
    axes[1,0].set_ylabel('MSE')
    axes[1,0].set_xticks(range(len(avg_mse)))
    axes[1,0].set_xticklabels(avg_mse.index, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, avg_mse.values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Average MAE across all horizons by dataset
    avg_mae = df.groupby('standard_dataset')['mae'].mean().sort_values()
    bars2 = axes[1,1].bar(range(len(avg_mae)), avg_mae.values, color='lightcoral', alpha=0.7)
    axes[1,1].set_title('Average MAE Across All Horizons')
    axes[1,1].set_xlabel('Dataset')
    axes[1,1].set_ylabel('MAE')
    axes[1,1].set_xticks(range(len(avg_mae)))
    axes[1,1].set_xticklabels(avg_mae.index, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, avg_mae.values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def generate_latex_table_all_horizons(df):
    """Generate LaTeX table with all forecast horizons"""
    
    # Baseline results from the template (these are example values)
    baseline_results = {
        'ETTm1': {'PDF': (0.342, 0.376), 'iTransformer': (0.347, 0.378), 'Pathformer': (0.357, 0.375), 
                  'FITS': (0.357, 0.377), 'TimeMixer': (0.356, 0.380), 'PatchTST': (0.349, 0.381)},
        'ETTm2': {'PDF': (0.250, 0.313), 'iTransformer': (0.258, 0.318), 'Pathformer': (0.253, 0.309), 
                  'FITS': (0.254, 0.313), 'TimeMixer': (0.257, 0.318), 'PatchTST': (0.256, 0.314)},
        'ETTh1': {'PDF': (0.407, 0.426), 'iTransformer': (0.440, 0.445), 'Pathformer': (0.417, 0.426), 
                  'FITS': (0.408, 0.427), 'TimeMixer': (0.427, 0.441), 'PatchTST': (0.419, 0.436)},
        'ETTh2': {'PDF': (0.347, 0.391), 'iTransformer': (0.359, 0.396), 'Pathformer': (0.360, 0.395), 
                  'FITS': (0.335, 0.386), 'TimeMixer': (0.347, 0.394), 'PatchTST': (0.351, 0.395)},
        'Traffic': {'PDF': (0.395, 0.270), 'iTransformer': (0.397, 0.281), 'Pathformer': (0.416, 0.264), 
                    'FITS': (0.429, 0.302), 'TimeMixer': (0.410, 0.279), 'PatchTST': (0.397, 0.275)},
        'Weather': {'PDF': (0.227, 0.263), 'iTransformer': (0.232, 0.270), 'Pathformer': (0.225, 0.258), 
                    'FITS': (0.244, 0.281), 'TimeMixer': (0.225, 0.263), 'PatchTST': (0.224, 0.261)},
        'Exchange': {'PDF': (0.350, 0.397), 'iTransformer': (0.321, 0.384), 'Pathformer': (0.384, 0.414), 
                     'FITS': (0.349, 0.396), 'TimeMixer': (0.385, 0.418), 'PatchTST': (0.322, 0.385)},
        'Electricity': {'PDF': (0.160, 0.253), 'iTransformer': (0.163, 0.258), 'Pathformer': (0.168, 0.261), 
                        'FITS': (0.169, 0.265), 'TimeMixer': (0.185, 0.284), 'PatchTST': (0.171, 0.270)}
    }
    
    # Get our results
    our_results = {}
    for dataset in df['standard_dataset'].unique():
        if pd.isna(dataset):
            continue
        dataset_df = df[df['standard_dataset'] == dataset]
        horizons_data = []
        for horizon in sorted(dataset_df['horizon'].unique()):
            horizon_df = dataset_df[dataset_df['horizon'] == horizon]
            if len(horizon_df) > 0:
                mse = horizon_df['mse'].iloc[0]
                mae = horizon_df['mae'].iloc[0]
                horizons_data.append(f"{mse:.3f} / {mae:.3f}")
        our_results[dataset] = horizons_data
    
    # Generate LaTeX table
    latex_code = """\\begin{table*}[ht]
\\centering
\\caption{Performance comparison (MSE / MAE) across datasets and models for all forecast horizons.}
\\renewcommand{\\arraystretch}{1.2}
\\setlength{\\tabcolsep}{3pt}
\\begin{adjustbox}{max width=\\textwidth}
\\begin{tabular}{l""" + "c" * (6 * 2 + len(max(our_results.values(), key=len))) + """}
\\toprule
\\multirow{2}{*}{Dataset} & 
\\multicolumn{2}{c}{\\textbf{PDF (2024)}} & 
\\multicolumn{2}{c}{\\textbf{iTransformer (2024)}} & 
\\multicolumn{2}{c}{\\textbf{Pathformer (2024)}} & 
\\multicolumn{2}{c}{\\textbf{FITS (2024)}} & 
\\multicolumn{2}{c}{\\textbf{TimeMixer (2024)}} & 
\\multicolumn{2}{c}{\\textbf{PatchTST (2023)}} & """
    
    # Add our method columns based on number of horizons
    max_horizons = max(len(horizons) for horizons in our_results.values())
    for i in range(max_horizons):
        latex_code += f"\\multicolumn{{1}}{{c}}{{\\textbf{{Our H{[96,192,336,720][i] if i < 4 else f'{i+1}'}}}}} & "
    
    latex_code = latex_code.rstrip(" & ") + " \\\\\n"
    
    # Add subheaders
    latex_code += " & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE"
    for i in range(max_horizons):
        latex_code += " & MSE/MAE"
    latex_code += " \\\\\n\\midrule\n"
    
    # Add data rows
    dataset_order = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Traffic', 'Weather', 'Exchange', 'Electricity']
    
    for dataset in dataset_order:
        if dataset in baseline_results and dataset in our_results:
            row = f"{dataset}"
            
            # Add baseline results
            for method in ['PDF', 'iTransformer', 'Pathformer', 'FITS', 'TimeMixer', 'PatchTST']:
                if method in baseline_results[dataset]:
                    mse, mae = baseline_results[dataset][method]
                    row += f" & {mse:.3f} & {mae:.3f}"
                else:
                    row += " & -- & --"
            
            # Add our results
            for horizon_result in our_results[dataset]:
                row += f" & {horizon_result}"
            
            # Pad with empty cells if needed
            for i in range(max_horizons - len(our_results[dataset])):
                row += " & --"
            
            row += " \\\\\n"
            latex_code += row
    
    latex_code += """\\bottomrule
\\end{tabular}
\\end{adjustbox}
\\label{tab:comparison_all_horizons}
\\end{table*}"""
    
    return latex_code

def generate_latex_table_mean(df):
    """Generate LaTeX table with mean values across all horizons"""
    
    # Baseline results from the template
    baseline_results = {
        'ETTm1': {'PDF': (0.342, 0.376), 'iTransformer': (0.347, 0.378), 'Pathformer': (0.357, 0.375), 
                  'FITS': (0.357, 0.377), 'TimeMixer': (0.356, 0.380), 'PatchTST': (0.349, 0.381)},
        'ETTm2': {'PDF': (0.250, 0.313), 'iTransformer': (0.258, 0.318), 'Pathformer': (0.253, 0.309), 
                  'FITS': (0.254, 0.313), 'TimeMixer': (0.257, 0.318), 'PatchTST': (0.256, 0.314)},
        'ETTh1': {'PDF': (0.407, 0.426), 'iTransformer': (0.440, 0.445), 'Pathformer': (0.417, 0.426), 
                  'FITS': (0.408, 0.427), 'TimeMixer': (0.427, 0.441), 'PatchTST': (0.419, 0.436)},
        'ETTh2': {'PDF': (0.347, 0.391), 'iTransformer': (0.359, 0.396), 'Pathformer': (0.360, 0.395), 
                  'FITS': (0.335, 0.386), 'TimeMixer': (0.347, 0.394), 'PatchTST': (0.351, 0.395)},
        'Traffic': {'PDF': (0.395, 0.270), 'iTransformer': (0.397, 0.281), 'Pathformer': (0.416, 0.264), 
                    'FITS': (0.429, 0.302), 'TimeMixer': (0.410, 0.279), 'PatchTST': (0.397, 0.275)},
        'Weather': {'PDF': (0.227, 0.263), 'iTransformer': (0.232, 0.270), 'Pathformer': (0.225, 0.258), 
                    'FITS': (0.244, 0.281), 'TimeMixer': (0.225, 0.263), 'PatchTST': (0.224, 0.261)},
        'Exchange': {'PDF': (0.350, 0.397), 'iTransformer': (0.321, 0.384), 'Pathformer': (0.384, 0.414), 
                     'FITS': (0.349, 0.396), 'TimeMixer': (0.385, 0.418), 'PatchTST': (0.322, 0.385)},
        'Electricity': {'PDF': (0.160, 0.253), 'iTransformer': (0.163, 0.258), 'Pathformer': (0.168, 0.261), 
                        'FITS': (0.169, 0.265), 'TimeMixer': (0.185, 0.284), 'PatchTST': (0.171, 0.270)}
    }
    
    # Calculate mean values for our results
    our_results = {}
    for dataset in df['standard_dataset'].unique():
        if pd.isna(dataset):
            continue
        dataset_df = df[df['standard_dataset'] == dataset]
        mean_mse = dataset_df['mse'].mean()
        mean_mae = dataset_df['mae'].mean()
        our_results[dataset] = (mean_mse, mean_mae)
    
    # Generate LaTeX table
    latex_code = """\\begin{table*}[ht]
\\centering
\\caption{Performance comparison (MSE / MAE) across datasets and models - Mean across all forecast horizons.}
\\renewcommand{\\arraystretch}{1.2}
\\setlength{\\tabcolsep}{4pt}
\\begin{adjustbox}{max width=\\textwidth}
\\begin{tabular}{lcccccccccccccc}
\\toprule
\\multirow{2}{*}{Dataset} & 
\\multicolumn{2}{c}{\\textbf{PDF (2024)}} & 
\\multicolumn{2}{c}{\\textbf{iTransformer (2024)}} & 
\\multicolumn{2}{c}{\\textbf{Pathformer (2024)}} & 
\\multicolumn{2}{c}{\\textbf{FITS (2024)}} & 
\\multicolumn{2}{c}{\\textbf{TimeMixer (2024)}} & 
\\multicolumn{2}{c}{\\textbf{PatchTST (2023)}} & 
\\multicolumn{2}{c}{\\textbf{Our}} \\\\
 & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\\\
\\midrule
"""
    
    # Add data rows
    dataset_order = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Traffic', 'Weather', 'Exchange', 'Electricity']
    
    for dataset in dataset_order:
        if dataset in baseline_results and dataset in our_results:
            row = f"{dataset}"
            
            # Add baseline results
            for method in ['PDF', 'iTransformer', 'Pathformer', 'FITS', 'TimeMixer', 'PatchTST']:
                if method in baseline_results[dataset]:
                    mse, mae = baseline_results[dataset][method]
                    row += f" & {mse:.3f} & {mae:.3f}"
                else:
                    row += " & -- & --"
            
            # Add our results
            our_mse, our_mae = our_results[dataset]
            row += f" & {our_mse:.3f} & {our_mae:.3f}"
            
            row += " \\\\\n"
            latex_code += row
    
    latex_code += """\\bottomrule
\\end{tabular}
\\end{adjustbox}
\\label{tab:comparison_mean}
\\end{table*}"""
    
    return latex_code

def main():
    """Main function to run the analysis"""
    
    # Load data
    csv_path = "c:/WUR/ssm_time_series/results/forecast_results_emb_128_tgt_1_20251108_111034.csv"
    df = load_and_process_data(csv_path)
    
    print("Data Overview:")
    print(f"Datasets: {df['standard_dataset'].unique()}")
    print(f"Horizons: {sorted(df['horizon'].unique())}")
    print(f"Total records: {len(df)}")
    print("\nSample data:")
    print(df[['standard_dataset', 'horizon', 'mse', 'mae']].head(10))
    
    # Create output directory
    output_dir = Path("c:/WUR/ssm_time_series/analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create and save plots
    fig = create_metrics_plot(df)
    plot_path = output_dir / "metrics_visualization.png"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.show()
    
    # Generate LaTeX tables
    latex_all = generate_latex_table_all_horizons(df)
    latex_mean = generate_latex_table_mean(df)
    
    # Save LaTeX tables
    with open(output_dir / "latex_table_all_horizons.tex", "w") as f:
        f.write(latex_all)
    
    with open(output_dir / "latex_table_mean.tex", "w") as f:
        f.write(latex_mean)
    
    print(f"\nLaTeX tables saved to:")
    print(f"- All horizons: {output_dir / 'latex_table_all_horizons.tex'}")
    print(f"- Mean values: {output_dir / 'latex_table_mean.tex'}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    summary = df.groupby('standard_dataset')[['mse', 'mae']].agg(['mean', 'std', 'min', 'max'])
    print("\nMean performance across all horizons:")
    for dataset in summary.index:
        mse_mean = summary.loc[dataset, ('mse', 'mean')]
        mae_mean = summary.loc[dataset, ('mae', 'mean')]
        print(f"{dataset:12} - MSE: {mse_mean:.4f}, MAE: {mae_mean:.4f}")
    
    print(f"\nBest performing dataset (lowest mean MSE): {summary[('mse', 'mean')].idxmin()}")
    print(f"Best performing dataset (lowest mean MAE): {summary[('mae', 'mean')].idxmin()}")
    
    # Print LaTeX tables to console
    print("\n" + "="*80)
    print("LATEX TABLE - ALL HORIZONS")
    print("="*80)
    print(latex_all)
    
    print("\n" + "="*80)
    print("LATEX TABLE - MEAN VALUES")
    print("="*80)
    print(latex_mean)

if __name__ == "__main__":
    main()