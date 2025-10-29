"""Evaluate a trained forecasting model across multiple datasets and create comparison table."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

# Ensure sibling packages under src are importable when running as a script from subdirectories
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_provider.utils.metrics import metric as compute_metric_values
from models.mamba_visual_encoder import MambaVisualEncoder
from util import build_time_series_dataloaders, default_device, prepare_run_directory


class ForecastModel(nn.Module):
    """Forecasting head that mirrors the training script architecture."""

    def __init__(
        self,
        encoder: MambaVisualEncoder,
        forecast_len: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.forecast_len = forecast_len
        self.forecast_head = nn.Linear(encoder.embedding_dim, forecast_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return self.forecast_head(embeddings)


def compute_metrics(predictions: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
    """Aggregate numpy predictions/targets into standard forecasting metrics."""

    if not predictions:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "mspe": float("nan"),
            "mse_loss": float("nan"),
            "num_samples": 0,
        }

    preds = np.concatenate(predictions, axis=0)
    trues = np.concatenate(targets, axis=0)
    mae, mse, rmse, mape, mspe = compute_metric_values(preds, trues)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "mse_loss": float(mse),
        "num_samples": int(preds.shape[0]),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Run forward passes over the dataloader and compute metrics."""

    model.eval()
    criterion = nn.MSELoss()
    batch_losses: List[float] = []
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            seq_x, seq_y, _, _ = batch
            seq_x = seq_x.to(device).float().transpose(1, 2)
            seq_y = seq_y.to(device).float()
            target = seq_y.reshape(seq_y.size(0), -1)

            predictions = model(seq_x)
            loss = criterion(predictions, target)

            batch_losses.append(loss.item())
            preds_list.append(predictions.detach().cpu().numpy())
            targets_list.append(target.detach().cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = compute_metrics(preds_list, targets_list)
    if batch_losses:
        metrics["mse_loss"] = float(np.mean(batch_losses))

    preds_array = np.concatenate(preds_list, axis=0) if preds_list else np.empty((0, 0), dtype=np.float32)
    targets_array = (
        np.concatenate(targets_list, axis=0) if targets_list else np.empty((0, 0), dtype=np.float32)
    )
    return metrics, preds_array, targets_array


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    """Load model weights and return the checkpoint dictionary."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def get_dataset_configs():
    """Define dataset configurations for evaluation."""
    return {
        "ETTm1": {
            "data_dir": "ICML_datasets",
            "filename": "/ETT-small/ETTm1.csv",
            "dataset_name": "ETTm1"
        },
        "ETTm2": {
            "data_dir": "ICML_datasets", 
            "filename": "/ETT-small/ETTm2.csv",
            "dataset_name": "ETTm2"
        },
        "ETTh1": {
            "data_dir": "ICML_datasets",
            "filename": "/ETT-small/ETTh1.csv", 
            "dataset_name": "ETTh1"
        },
        "ETTh2": {
            "data_dir": "ICML_datasets",
            "filename": "/ETT-small/ETTh2.csv",
            "dataset_name": "ETTh2"
        },
        "Traffic": {
            "data_dir": "ICML_datasets",
            "filename": "/traffic/traffic.csv",
            "dataset_name": "traffic"
        },
        "Weather": {
            "data_dir": "ICML_datasets", 
            "filename": "/weather/weather.csv",
            "dataset_name": "weather"
        },
        "Exchange": {
            "data_dir": "ICML_datasets",
            "filename": "/exchange_rate/exchange_rate.csv",
            "dataset_name": "exchange_rate"
        },
        "Solar": {
            "data_dir": "ICML_datasets",
            "filename": "/Solar/solar_AL.csv",
            "dataset_name": "Solar"
        },
        "Electricity": {
            "data_dir": "ICML_datasets",
            "filename": "/electricity/electricity.csv", 
            "dataset_name": "electricity"
        }
    }


def evaluate_single_dataset(
    dataset_name: str,
    dataset_config: Dict,
    model: nn.Module,
    args,
    device: torch.device,
    run_dir: Path
) -> Dict[str, float]:
    """Evaluate model on a single dataset and return metrics."""
    
    print(f"\n{'='*50}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*50}")
    
    # Build dataset-specific data directory path
    data_dir = Path(args.base_data_dir) / dataset_config["data_dir"]
    
    try:
        data_kwargs = dict(
            data_dir=str(data_dir),
            filename=dataset_config["filename"],
            dataset_name=dataset_config["dataset_name"],
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            train=True,
            val=True,
            test=args.split == "test",
        )

        loaders = build_time_series_dataloaders(**data_kwargs)
        if args.split == "test":
            train_loader, val_loader, test_loader = loaders
            eval_loader = test_loader
        else:
            train_loader, val_loader = loaders
            eval_loader = train_loader if args.split == "train" else val_loader

        if eval_loader is None:
            print(f"Warning: No {args.split} split available for {dataset_name}")
            return {"mae": float("nan"), "mse": float("nan"), "error": "No data available"}

        # Get forecast length from sample batch
        sample_batch = next(iter(eval_loader))
        sample_target = sample_batch[1]
        forecast_len = sample_target.size(1) * sample_target.size(2)
        
        # Recreate model with correct forecast length
        encoder = MambaVisualEncoder(
            input_dim=args.token_size,
            model_dim=args.model_dim,
            depth=args.depth,
            embedding_dim=args.embedding_dim,
            pooling="mean",
            dropout=0.1,
        )
        
        dataset_model = ForecastModel(encoder=encoder, forecast_len=forecast_len).to(device)
        
        # Load checkpoint
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        load_checkpoint(dataset_model, checkpoint_path, device)
        
        # Evaluate
        metrics, predictions, targets = evaluate_model(dataset_model, eval_loader, device)
        
        print(f"Results for {dataset_name}:")
        print(f"  MSE: {metrics['mse']:.3f}")
        print(f"  MAE: {metrics['mae']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  Samples: {metrics['num_samples']}")
        
        # Save individual dataset results
        dataset_run_dir = run_dir / f"{dataset_name.lower()}"
        dataset_run_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_payload = {
            "dataset": dataset_name,
            "checkpoint_path": str(checkpoint_path),
            "split": args.split,
            "metrics": metrics,
            "config": dataset_config,
        }
        
        with (dataset_run_dir / "metrics.json").open("w") as f:
            json.dump(metrics_payload, f, indent=2)
        
        # Save predictions and targets
        tensor_payload = {
            "predictions": torch.from_numpy(predictions.astype(np.float32, copy=False)),
            "targets": torch.from_numpy(targets.astype(np.float32, copy=False)),
            "metrics": metrics,
            "dataset": dataset_name,
        }
        torch.save(tensor_payload, dataset_run_dir / "evaluation.pt")
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {dataset_name}: {str(e)}")
        return {"mae": float("nan"), "mse": float("nan"), "error": str(e)}


def create_results_table(results: Dict[str, Dict[str, float]], model_name: str = "MambaEncoder") -> pd.DataFrame:
    """Create a formatted results table similar to the reference format."""
    
    # Create the table structure
    datasets = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Traffic", "Weather", "Exchange", "Solar", "Electricity"]
    
    # Initialize results dictionary
    table_data = {
        "Dataset": datasets,
        f"{model_name}_MSE": [],
        f"{model_name}_MAE": [],
    }
    
    # Fill in the results
    for dataset in datasets:
        if dataset in results and not np.isnan(results[dataset].get("mse", np.nan)):
            table_data[f"{model_name}_MSE"].append(f"{results[dataset]['mse']:.3f}")
            table_data[f"{model_name}_MAE"].append(f"{results[dataset]['mae']:.3f}")
        else:
            table_data[f"{model_name}_MSE"].append("N/A")
            table_data[f"{model_name}_MAE"].append("N/A")
    
    df = pd.DataFrame(table_data)
    return df


def main() -> None:
    # Parse arguments
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=r"C:\WUR\ssm_time_series\src\configs\forecast_evaluation.yaml",
        help="Path to YAML config describing CLI args",
    )
    known, _ = pre_parser.parse_known_args()
    
    # Load config
    config_str = os.path.expanduser(os.path.expandvars(known.config))
    config_norm = os.path.normpath(config_str)
    config_path = Path(config_norm).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        schema = yaml.safe_load(f) or {}

    # Build argument parser
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": lambda s: s.lower() in ("1", "true", "yes"),
    }

    parser = argparse.ArgumentParser(description="Evaluate forecasting model across multiple datasets")
    parser.add_argument("--config", type=str, default=str(config_path))
    parser.add_argument("--model_name", type=str, default="MambaEncoder", help="Name for the model in results table")
    parser.add_argument("--base_data_dir", type=str, default="C:/WUR/ssm_time_series", help="Base directory containing ICML_datasets")
    
    for name, spec in schema.items():
        arg_name = f"--{name}"
        arg_type = type_map.get(spec.get("type", "str"), str)
        default = spec.get("default", None)
        required = bool(spec.get("required", False))
        help_text = spec.get("description", None)
        choices = spec.get("choices", None)
        
        parser.add_argument(
            arg_name,
            type=arg_type,
            default=default,
            required=required and default is None,
            help=help_text,
            choices=choices,
        )

    args = parser.parse_args()

    device = default_device()
    print(f"Using device: {device}")
    print(f"Model name: {args.model_name}")
    print(f"Base data directory: {args.base_data_dir}")

    # Create output directory
    run_dir = prepare_run_directory(Path(args.output_dir), f"multi_dataset_eval_{args.split}")
    print(f"Saving evaluation artifacts to: {run_dir}")

    # Get dataset configurations
    dataset_configs = get_dataset_configs()
    
    # Results storage
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name, dataset_config in dataset_configs.items():
        print(f"\nStarting evaluation for {dataset_name}...")
        
        # Create a dummy model to get the architecture right (will be reloaded for each dataset)
        encoder = MambaVisualEncoder(
            input_dim=args.token_size,
            model_dim=args.model_dim,
            depth=args.depth,
            embedding_dim=args.embedding_dim,
            pooling="mean",
            dropout=0.1,
        )
        dummy_model = ForecastModel(encoder=encoder, forecast_len=96).to(device)  # Dummy forecast_len
        
        # Evaluate this dataset
        metrics = evaluate_single_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            model=dummy_model,
            args=args,
            device=device,
            run_dir=run_dir
        )
        
        all_results[dataset_name] = metrics

    # Create and save results table
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    results_df = create_results_table(all_results, args.model_name)
    print("\nResults Table:")
    print(results_df.to_string(index=False))
    
    # Save results table
    results_df.to_csv(run_dir / "results_table.csv", index=False)
    results_df.to_excel(run_dir / "results_table.xlsx", index=False)
    
    # Save complete results as JSON
    summary_results = {
        "model_name": args.model_name,
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "results_by_dataset": all_results,
        "summary_stats": {
            "avg_mse": np.nanmean([r.get("mse", np.nan) for r in all_results.values()]),
            "avg_mae": np.nanmean([r.get("mae", np.nan) for r in all_results.values()]),
            "datasets_evaluated": len([r for r in all_results.values() if not np.isnan(r.get("mse", np.nan))]),
            "total_datasets": len(all_results),
        }
    }
    
    with (run_dir / "complete_results.json").open("w") as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    print(f"\nSummary Statistics:")
    print(f"  Average MSE: {summary_results['summary_stats']['avg_mse']:.3f}")
    print(f"  Average MAE: {summary_results['summary_stats']['avg_mae']:.3f}")
    print(f"  Datasets Successfully Evaluated: {summary_results['summary_stats']['datasets_evaluated']}/{summary_results['summary_stats']['total_datasets']}")
    
    print(f"\nAll results saved to: {run_dir}")
    print("Files created:")
    print("  - results_table.csv/xlsx: Formatted comparison table")
    print("  - complete_results.json: Full results with metadata")
    print("  - [dataset_name]/: Individual dataset results and predictions")


if __name__ == "__main__":
    main()

# Example usage:
# python -m evaluation.multi_dataset_evaluation \
#   --config configs/forecast_evaluation.yaml \
#   --checkpoint_path "C:\WUR\ssm_time_series\checkpoints\forecast_encoder_20251014_2306\best.pt" \
#   --split test \
#   --output_dir "C:\WUR\ssm_time_series\eval_runs" \
#   --model_name "MambaEncoder" \
#   --base_data_dir "C:\WUR\ssm_time_series"