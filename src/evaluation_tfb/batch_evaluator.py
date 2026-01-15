import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import argparse
import traceback
from typing import List, Optional, Dict

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation_tfb.dataloader import get_tfb_dataloader
from evaluation_tfb.model_wrapper import load_dual_encoder_model
from evaluation_tfb.evaluator import TFBEvaluator

def save_results_csv(all_results: Dict[str, Dict[int, Dict[str, float]]], output_file: Path):
    rows = []
    for ds_name, horizons_data in all_results.items():
        for horizon, metrics in horizons_data.items():
            rows.append({
                "Dataset": ds_name,
                "Horizon": horizon,
                "MSE": metrics["mse"],
                "MAE": metrics["mae"]
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"CSV results saved to {output_file}")

def save_results_latex(all_results: Dict[str, Dict[int, Dict[str, float]]], output_file: Path, horizons: List[int]):
    with open(output_file, "w") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{TFB Batch Evaluation Results}\n")
        
        # Column definition: Dataset, then (MSE, MAE) pairs for each horizon
        col_def = "l" + "cc" * len(horizons)
        f.write(f"\\begin{{tabular}}{{{col_def}}}\n")
        f.write("\\toprule\n")
        
        # Header Row 1: Horizons
        f.write("Dataset")
        for h in horizons:
            f.write(f" & \\multicolumn{{2}}{{c}}{{Horizon {h}}}")
        f.write(" \\\\\n")
        
        # Header Row 2: Metrics
        for h in horizons:
            f.write(" & MSE & MAE")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        
        # Data Rows
        datasets = sorted(all_results.keys())
        for ds in datasets:
            # Clean dataset name for LaTeX (replace underscores)
            ds_clean = ds.replace("_", "\\_")
            line = f"{ds_clean}"
            for h in horizons:
                if h in all_results[ds]:
                    mse = all_results[ds][h]["mse"]
                    mae = all_results[ds][h]["mae"]
                    line += f" & {mse:.4f} & {mae:.4f}"
                else:
                    line += " & - & -"
            f.write(line + " \\\\\n")
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"LaTeX results saved to {output_file}")

def run_batch_evaluation(
    data_dir: str,
    model_config: str,
    forecast_checkpoint: str,
    encoder_checkpoint: Optional[str] = None,
    visual_encoder_checkpoint: Optional[str] = None,
    horizons: List[int] = [96, 192, 336, 720],
    results_dir: str = "results/tfb_batch",
    device: str = "auto",
    is_meta_config: bool = False,
    stride: int = 1,
    batch_size: int = 32,
    max_datasets: Optional[int] = None,
):
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model...")
    model, _, eval_horizons, max_horizon = load_dual_encoder_model(
        model_config_path=model_config,
        forecast_checkpoint_path=forecast_checkpoint,
        encoder_checkpoint_path=encoder_checkpoint,
        visual_encoder_checkpoint_path=visual_encoder_checkpoint,
        requested_horizons=horizons,
        device=device,
        is_meta_config=is_meta_config
    )
    
    # Initialize evaluator
    device_obj = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    evaluator = TFBEvaluator(
        model=model,
        device=device_obj,
        eval_horizons=eval_horizons,
        max_horizon=max_horizon
    )
    
    # Discover datasets
    data_path = Path(data_dir)
    dataset_files = sorted([f.name for f in data_path.glob("*.csv") if f.name != "FORECAST_META.csv"])
    
    if max_datasets and max_datasets > 0:
        dataset_files = dataset_files[:max_datasets]
        print(f"Limiting to first {max_datasets} datasets.")
    
    print(f"Found {len(dataset_files)} datasets in {data_dir}")
    
    all_results = {}
    
    for ds_file in tqdm(dataset_files, desc="Batch Evaluation"):
        ds_name = ds_file
        file_path = str(data_path / ds_file)
        
        try:
            # Get dataloader
            dataloader, dataset = get_tfb_dataloader(
                file_path=file_path,
                batch_size=batch_size,
                context_length=96,
                prediction_length=max_horizon,
                stride=stride,
                num_workers=0 # Safer for batch processing on Windows
            )
            
            if len(dataset) == 0:
                print(f"  [Warning] Skipping {ds_name}: No windows found.")
                continue
                
            # Evaluate
            ds_metrics = evaluator.evaluate(
                dataloader=dataloader,
                dataset_name=ds_name,
                inverse_transform_fn=dataset.inverse_transform
            )
            
            all_results[ds_name] = ds_metrics
            
        except Exception as e:
            print(f"  [Error] Failed to evaluate {ds_name}: {e}")
            # traceback.print_exc()

    # Save to CSV
    save_results_csv(all_results, results_path / "tfb_batch_results.csv")
    
    # Generate LaTeX
    save_results_latex(all_results, results_path / "tfb_batch_results.tex", eval_horizons)
    
    print(f"Batch evaluation complete. Results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFB Batch Evaluation Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TFB forecasting CSVs")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--forecast_checkpoint", type=str, required=True, help="Path to forecast head checkpoint")
    parser.add_argument("--encoder_checkpoint", type=str, help="Path to encoder checkpoint")
    parser.add_argument("--visual_encoder_checkpoint", type=str, help="Path to visual encoder checkpoint")
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720], help="Forecast horizons")
    parser.add_argument("--results_dir", type=str, default="results/tfb_batch", help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--is_meta_config", action="store_true", help="Whether model_config is a meta-config")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sliding window")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max_datasets", type=int, help="Maximum number of datasets to evaluate")
    
    args = parser.parse_args()
    
    run_batch_evaluation(
        data_dir=args.data_dir,
        model_config=args.model_config,
        forecast_checkpoint=args.forecast_checkpoint,
        encoder_checkpoint=args.encoder_checkpoint,
        visual_encoder_checkpoint=args.visual_encoder_checkpoint,
        horizons=args.horizons,
        results_dir=args.results_dir,
        device=args.device,
        is_meta_config=args.is_meta_config,
        stride=args.stride,
        batch_size=args.batch_size,
        max_datasets=args.max_datasets
    )
