"""Create comprehensive comparison tables from multiple model evaluation results."""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def load_evaluation_results(results_dir: Path) -> Dict[str, Any]:
    """Load evaluation results from a complete_results.json file."""
    results_file = results_dir / "complete_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with results_file.open("r") as f:
        return json.load(f)


def create_comparison_table(
    model_results: Dict[str, Dict[str, Any]], 
    datasets: List[str] = None,
    metrics: List[str] = ["MSE", "MAE"]
) -> pd.DataFrame:
    """Create a comparison table similar to the reference format."""
    
    if datasets is None:
        datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Traffic", "Weather", "Exchange", "Solar", "Electricity"]
    
    # Prepare data structure
    table_data = {"Dataset": datasets}
    
    # Create columns for each model and metric
    for model_name in model_results.keys():
        for metric in metrics:
            table_data[f"{model_name}_{metric}"] = []
    
    # Fill in the data
    for dataset in datasets:
        for model_name, results in model_results.items():
            dataset_results = results.get("results_by_dataset", {}).get(dataset, {})
            
            for metric in metrics:
                metric_key = metric.lower()
                if metric_key in dataset_results and not np.isnan(dataset_results[metric_key]):
                    table_data[f"{model_name}_{metric}"].append(f"{dataset_results[metric_key]:.3f}")
                else:
                    table_data[f"{model_name}_{metric}"].append("N/A")
    
    df = pd.DataFrame(table_data)
    return df


def create_model_summary_table(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary table with average metrics across all datasets."""
    
    summary_data = {
        "Model": [],
        "Avg_MSE": [],
        "Avg_MAE": [],
        "Datasets_Evaluated": [],
        "Total_Datasets": []
    }
    
    for model_name, results in model_results.items():
        summary_stats = results.get("summary_stats", {})
        
        summary_data["Model"].append(model_name)
        summary_data["Avg_MSE"].append(f"{summary_stats.get('avg_mse', np.nan):.3f}")
        summary_data["Avg_MAE"].append(f"{summary_stats.get('avg_mae', np.nan):.3f}")
        summary_data["Datasets_Evaluated"].append(summary_stats.get('datasets_evaluated', 0))
        summary_data["Total_Datasets"].append(summary_stats.get('total_datasets', 0))
    
    return pd.DataFrame(summary_data)


def find_best_results(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Find the best performing model for each dataset and metric."""
    
    datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Traffic", "Weather", "Exchange", "Solar", "Electricity"]
    metrics = ["mse", "mae"]
    
    best_results = {}
    
    for dataset in datasets:
        best_results[dataset] = {}
        
        for metric in metrics:
            best_value = float('inf')
            best_model = "N/A"
            
            for model_name, results in model_results.items():
                dataset_results = results.get("results_by_dataset", {}).get(dataset, {})
                
                if metric in dataset_results and not np.isnan(dataset_results[metric]):
                    value = dataset_results[metric]
                    if value < best_value:
                        best_value = value
                        best_model = model_name
            
            best_results[dataset][metric] = best_model if best_model != "N/A" else "N/A"
    
    return best_results


def main():
    parser = argparse.ArgumentParser(description="Create comparison tables from multiple model evaluations")
    parser.add_argument("--results_dirs", nargs="+", required=True, 
                       help="Paths to evaluation result directories")
    parser.add_argument("--model_names", nargs="+", 
                       help="Optional custom names for models (must match order of results_dirs)")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                       help="Directory to save comparison tables")
    parser.add_argument("--include_best_summary", action="store_true",
                       help="Include a summary of best performing models per dataset")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results from all directories
    model_results = {}
    
    for i, results_dir in enumerate(args.results_dirs):
        results_path = Path(results_dir)
        
        try:
            results = load_evaluation_results(results_path)
            
            # Use custom name if provided, otherwise use model_name from results
            if args.model_names and i < len(args.model_names):
                model_name = args.model_names[i]
            else:
                model_name = results.get("model_name", f"Model_{i+1}")
            
            model_results[model_name] = results
            print(f"Loaded results for {model_name} from {results_path}")
            
        except Exception as e:
            print(f"Warning: Could not load results from {results_dir}: {e}")
            continue
    
    if not model_results:
        print("Error: No valid results found!")
        return
    
    print(f"\nCreating comparison tables for {len(model_results)} models...")
    
    # Create main comparison table
    comparison_df = create_comparison_table(model_results)
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    comparison_df.to_excel(output_dir / "model_comparison.xlsx", index=False)
    
    # Create summary table
    summary_df = create_model_summary_table(model_results)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    # Save summary table
    summary_df.to_csv(output_dir / "model_summary.csv", index=False)
    summary_df.to_excel(output_dir / "model_summary.xlsx", index=False)
    
    # Create best results summary if requested
    if args.include_best_summary:
        best_results = find_best_results(model_results)
        
        # Convert to DataFrame for better visualization
        best_df_data = {"Dataset": []}
        for metric in ["MSE", "MAE"]:
            best_df_data[f"Best_{metric}"] = []
        
        datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Traffic", "Weather", "Exchange", "Solar", "Electricity"]
        for dataset in datasets:
            best_df_data["Dataset"].append(dataset)
            best_df_data["Best_MSE"].append(best_results[dataset]["mse"])
            best_df_data["Best_MAE"].append(best_results[dataset]["mae"])
        
        best_df = pd.DataFrame(best_df_data)
        print("\nBest Performing Models per Dataset:")
        print(best_df.to_string(index=False))
        
        best_df.to_csv(output_dir / "best_models_per_dataset.csv", index=False)
        best_df.to_excel(output_dir / "best_models_per_dataset.xlsx", index=False)
    
    # Create a formatted table similar to the reference
    print(f"\n{'='*80}")
    print("FORMATTED COMPARISON TABLE (Reference Style)")
    print(f"{'='*80}")
    
    # Restructure for reference-style table
    datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Traffic", "Weather", "Exchange", "Solar", "Electricity"]
    
    print("Models", end="")
    for model_name in model_results.keys():
        print(f"\t{model_name}", end="")
    print()
    
    print("Metric", end="")
    for model_name in model_results.keys():
        print(f"\tMSE\tMAE", end="")
    print()
    
    for dataset in datasets:
        print(f"{dataset}", end="")
        for model_name in model_results.keys():
            dataset_results = model_results[model_name].get("results_by_dataset", {}).get(dataset, {})
            mse = dataset_results.get("mse", np.nan)
            mae = dataset_results.get("mae", np.nan)
            
            mse_str = f"{mse:.3f}" if not np.isnan(mse) else "N/A"
            mae_str = f"{mae:.3f}" if not np.isnan(mae) else "N/A"
            
            print(f"\t{mse_str}\t{mae_str}", end="")
        print()
    
    print(f"\nAll comparison tables saved to: {output_dir}")
    print("Files created:")
    print("  - model_comparison.csv/xlsx: Main comparison table")
    print("  - model_summary.csv/xlsx: Average performance summary")
    if args.include_best_summary:
        print("  - best_models_per_dataset.csv/xlsx: Best model per dataset")


if __name__ == "__main__":
    main()

# Example usage:
# python -m evaluation.create_comparison_table \
#   --results_dirs "eval_runs/multi_dataset_eval_test_20251029_1500" "eval_runs/other_model_eval_test_20251029_1600" \
#   --model_names "MambaEncoder" "OtherModel" \
#   --output_dir "comparison_results" \
#   --include_best_summary