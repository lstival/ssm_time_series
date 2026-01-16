import os
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

# Setup paths
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation_tfb.dataloader import get_tfb_dataloader
from evaluation_tfb.evaluator import TFBEvaluator
from evaluation_tfb.zoo_models import get_model
from evaluation_tfb.batch_evaluator import save_results_csv, save_results_latex

def run_multi_model_evaluation(
    data_dir: str,
    model_names: List[str],
    horizons: List[int] = [96, 192, 336, 720],
    results_root: str = "results/zero_shot_benchmarks",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    stride: int = 1,
    max_datasets: Optional[int] = None,
):
    """
    Evaluates multiple zero-shot models on all datasets in data_dir.
    """
    data_path = Path(data_dir)
    dataset_files = sorted([f.name for f in data_path.glob("*.csv") if f.name != "FORECAST_META.csv"])
    
    if max_datasets:
        dataset_files = dataset_files[:max_datasets]

    results_root_path = Path(results_root)
    results_root_path.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        print(f"\n{'='*20} Evaluating Model: {model_name} {'='*20}")
        
        # Load Model
        try:
            model = get_model(model_name, device=device)
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue

        # In zero-shot models, max_horizon is usually fixed or flexible
        eval_horizons = horizons
        max_horizon = max(horizons)

        evaluator = TFBEvaluator(
            model=model,
            device=torch.device(device),
            eval_horizons=eval_horizons,
            max_horizon=max_horizon
        )

        model_results = {}
        
        for ds_file in tqdm(dataset_files, desc=f"Benchmarking {model_name}"):
            file_path = str(data_path / ds_file)
            try:
                # Get dataloader
                dataloader, dataset = get_tfb_dataloader(
                    file_path=file_path,
                    batch_size=batch_size,
                    context_length=96, # Context length for benchmarks usually 96 or 512
                    prediction_length=max_horizon,
                    stride=stride,
                    num_workers=0 
                )
                
                if len(dataset) == 0:
                    continue
                
                # Evaluate (Evaluator handles the baseline offset and normalization)
                metrics = evaluator.evaluate(
                    dataloader=dataloader,
                    dataset_name=ds_file,
                    inverse_transform_fn=None # Evaluation before denormalization as requested
                )
                
                model_results[ds_file] = metrics
                
            except Exception as e:
                print(f"Error evaluating {ds_file} with {model_name}: {e}")
                continue

        # Save results for this model
        model_safe_name = model_name.replace("/", "_").replace("-", "_")
        model_out_dir = results_root_path / model_safe_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        save_results_csv(model_results, model_out_dir / "results.csv")
        save_results_latex(model_results, model_out_dir / "results.tex", eval_horizons)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Zero-Shot Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with CSV datasets")
    parser.add_argument("--models", type=str, nargs="+", default=["timesfm", "chronos", "transformer"], help="List of models to evaluate")
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    parser.add_argument("--results_dir", type=str, default="results/zero_shot_benchmarks")
    parser.add_argument("--max_datasets", type=int, help="Limit number of datasets")
    
    args = parser.parse_args()
    
    run_multi_model_evaluation(
        data_dir=args.data_dir,
        model_names=args.models,
        horizons=args.horizons,
        results_root=args.results_dir,
        max_datasets=args.max_datasets
    )
