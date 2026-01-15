import argparse
import os
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import traceback
from typing import List, Optional

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# Also add project root for other imports
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_tfb.dataloader import get_tfb_dataloader
from evaluation_tfb.model_wrapper import load_dual_encoder_model
from evaluation_tfb.evaluator import TFBEvaluator, aggregate_tfb_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate forecasting models on TFB datasets.")
    
    # Model config
    parser.add_argument("--model_config", type=str, help="Path to model YAML config or Meta config.")
    parser.add_argument("--is_meta_config", action="store_true", help="Flag if model_config is a meta-config (like icml_zeroshot_dual.yaml).")
    
    parser.add_argument("--forecast_checkpoint", type=str, help="Path to forecast head checkpoint.")
    parser.add_argument("--encoder_checkpoint", type=str, help="Path to encoder checkpoint.")
    parser.add_argument("--visual_encoder_checkpoint", type=str, help="Path to visual encoder checkpoint.")
    
    # Data config
    parser.add_argument("--data_dir", type=str, default="data/forecasting/forecasting", help="Directory containing TFB CSVs.")
    parser.add_argument("--datasets", nargs="+", help="Specific CSV filenames to evaluate. If empty, all in data_dir will be used.")
    
    # Evaluation config
    parser.add_argument("--horizons", type=int, nargs="+", help="Evaluation horizons. (Defaults to config or [96, 192, 336, 720])")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--context_length", type=int, default=96, help="Context length for model input.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sliding window.")
    parser.add_argument("--results_dir", type=str, default="results/evaluation_tfb", help="Directory to save results.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu).")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup results dir
    results_path = Path(args.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Identify datasets
    data_dir = Path(args.data_dir)
    if args.datasets:
        dataset_files = [data_dir / d for d in args.datasets]
    else:
        dataset_files = list(data_dir.glob("*.csv"))
        # Exclude metadata file if present
        dataset_files = [f for f in dataset_files if f.name != "FORECAST_META.csv"]
        
    if not dataset_files:
        print(f"No datasets found in {args.data_dir}")
        return

    # Horizons defaults
    horizons = args.horizons if args.horizons else [96, 192, 336, 720]

    # Load model
    print(f"Loading model...")
    try:
        model, checkpoint_info, eval_horizons, max_horizon = load_dual_encoder_model(
            model_config_path=args.model_config,
            forecast_checkpoint_path=args.forecast_checkpoint,
            encoder_checkpoint_path=args.encoder_checkpoint,
            visual_encoder_checkpoint_path=args.visual_encoder_checkpoint,
            requested_horizons=horizons,
            device=args.device,
            is_meta_config=args.is_meta_config
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        traceback.print_exc()
        return
    
    # Initialize evaluator
    import torch
    evaluator = TFBEvaluator(
        model=model,
        device=torch.device(args.device),
        eval_horizons=eval_horizons,
        max_horizon=max_horizon
    )
    
    results_by_dataset = {}
    
    # Evaluate each dataset
    for dataset_file in dataset_files:
        if not dataset_file.exists():
            print(f"Dataset {dataset_file} not found, skipping.")
            continue
            
        print(f"Evaluating {dataset_file.name}...")
        try:
            dataloader, dataset_obj = get_tfb_dataloader(
                file_path=str(dataset_file),
                batch_size=args.batch_size,
                context_length=args.context_length,
                prediction_length=max_horizon,
                stride=args.stride
            )
            
            metrics = evaluator.evaluate(
                dataloader=dataloader,
                dataset_name=dataset_file.name,
                inverse_transform_fn=dataset_obj.inverse_transform
            )
            
            results_by_dataset[dataset_file.name] = metrics
            
            # Print current metrics
            for h in eval_horizons:
                m = metrics[h]
                print(f"  H{h}: MSE={m['mse']:.6f}, MAE={m['mae']:.6f}")
                
        except Exception as e:
            print(f"Error evaluating {dataset_file.name}: {e}")
            traceback.print_exc()
            continue
            
    # Aggregate results
    if not results_by_dataset:
        print("No metrics produced.")
        return
        
    summary = aggregate_tfb_results(results_by_dataset)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_base = results_path / f"tfb_eval_{timestamp}"
    
    with open(f"{output_base}_full.json", "w") as f:
        json.dump(results_by_dataset, f, indent=4)
        
    with open(f"{output_base}_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nEvaluation Summary:")
    for h in sorted(summary.keys()):
        s = summary[h]
        print(f"  H{h}: Mean MSE={s['mean_mse']:.6f}, Mean MAE={s['mean_mae']:.6f} ({s['dataset_count']} datasets)")
        
    print(f"\nResults saved to {args.results_dir}")

if __name__ == "__main__":
    main()
