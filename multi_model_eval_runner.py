import os
import argparse
import pandas as pd
import torch
import json
from datetime import datetime
from src.evaluation_tfb.dataloader import TFBDataset
from torch.utils.data import DataLoader
from src.evaluation_tfb.evaluator import TFBEvaluator
from src.evaluation_tfb.zoo_models import get_model

def main():
    parser = argparse.ArgumentParser(description="Multi-model Evaluation Runner for TFB")
    parser.add_argument("--data_dir", type=str, default="data/forecasting", help="Path to TFB datasets")
    parser.add_argument("--models", type=str, default="timesfm,chronos", help="Comma-separated model names (timesfm, chronos, transformer, patchtst)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="results/zoo_eval")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names to run (e.g. Bitcoin,Traffic). If None, runs all.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_names = [m.strip() for m in args.models.split(",")]
    
    # 1. Get list of datasets
    all_datasets = [f for f in os.listdir(args.data_dir) if f.endswith(".csv")]
    if args.datasets:
        target_datasets = [d.strip() for d in args.datasets.split(",")]
        all_datasets = [f for f in all_datasets if any(t in f for t in target_datasets)]
        
    print(f"Found {len(all_datasets)} datasets.")
    
    results_summary = []
    
    for model_name in model_names:
        print(f"\n{'='*20}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*20}")
        
        try:
            model = get_model(model_name, device=args.device)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue
            
        horizons = [96, 192, 336, 720]
        max_horizon = 720
        evaluator = TFBEvaluator(model, device=args.device, eval_horizons=horizons, max_horizon=max_horizon)
        
        for dataset_file in all_datasets:
            dataset_path = os.path.join(args.data_dir, dataset_file)
            dataset_name = dataset_file.replace(".csv", "")
            print(f"\nProcessing {dataset_name}...")
            
            try:
                dataset = TFBDataset(dataset_path)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                
                metrics = evaluator.evaluate(dataloader, dataset_name=dataset_name)
                
                # Save individual result
                result_entry = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "metrics": metrics
                }
                
                # Append to summary
                for h, vals in metrics.items():
                    results_summary.append({
                        "Model": model_name,
                        "Dataset": dataset_name,
                        "Horizon": h,
                        "MSE": vals["mse"],
                        "MAE": vals["mae"]
                    })
                
                # Save partial results to avoid data loss
                temp_df = pd.DataFrame(results_summary)
                temp_df.to_csv(os.path.join(args.output_dir, f"partial_results_{model_name}.csv"), index=False)
                
            except Exception as e:
                print(f"Error evaluating {dataset_name} with {model_name}: {e}")
                
        # Clean up memory
        del model
        torch.cuda.empty_cache()

    # Save final results
    final_df = pd.DataFrame(results_summary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv = os.path.join(args.output_dir, f"final_zoo_results_{timestamp}.csv")
    final_df.to_csv(final_csv, index=False)
    
    print(f"\nEvaluation complete. Results saved to {final_csv}")
    
    # Calculate averages per model
    if not final_df.empty:
        avg_results = final_df.groupby(["Model", "Horizon"])[["MSE", "MAE"]].mean().reset_index()
        print("\nAVERAGE RESULTS:")
        print(avg_results)
        avg_results.to_csv(os.path.join(args.output_dir, f"averaged_results_{timestamp}.csv"), index=False)

if __name__ == "__main__":
    main()
