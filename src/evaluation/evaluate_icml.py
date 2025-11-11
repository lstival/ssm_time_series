"""Evaluate trained multi-horizon forecast model on embedding cache datasets.

This variant loads embeddings directly from the embedding cache and evaluates
on the test split only. Configuration is provided via a dataclass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Add project paths
SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import numpy as np

from util import default_device
from down_tasks.forecast_utils import (
    load_trained_model,
    save_evaluation_results,
    print_evaluation_summary,
    plot_forecast_comparison,
    MultiHorizonForecastMLP
)
from dataloaders.embedding_cache_dataset import build_embedding_cache_loader
from tqdm import tqdm


@dataclass
class EvalConfig:
    checkpoint_path: str = r"C:\WUR\ssm_time_series\checkpoints\multi_horizon_forecast_emb_128_tgt_1_20251108_1124\best_model.pt"
    embedding_cache_dir: str = r"C:\WUR\ssm_time_series\embedding_cache"
    results_dir: str = r"C:\WUR\ssm_time_series\results"
    plots_dir: str = r"C:\WUR\ssm_time_series\plots"
    batch_size: int = 64
    num_workers: int = 4
    split: str = "test"  # one of "test", "val", "train"
    dataset_filter: Optional[List[str]] = None  # e.g., ["electricity", "weather"]
    horizons: Optional[str] = None  # comma-separated string, e.g., "96,192,336,720"
    generate_plots: bool = False
    samples_per_plot: int = 3
    show_plots: bool = False
    mlp_hidden_dim: Optional[int] = None


def get_default_config() -> EvalConfig:
    return EvalConfig()


def discover_embedding_datasets(embedding_cache_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover all dataset/subdataset combinations in the embedding cache directory.
    
    Returns:
        List of (dataset_group, dataset_name, dataset_path) tuples
    """
    datasets = []
    for group_dir in embedding_cache_dir.iterdir():
        if group_dir.is_dir():
            for dataset_dir in group_dir.iterdir():
                if dataset_dir.is_dir() and (dataset_dir / "metadata.json").exists():
                    datasets.append((group_dir.name, dataset_dir.name, dataset_dir))
    return datasets


def evaluate_embedding_dataset(
    model: MultiHorizonForecastMLP,
    dataset_path: Path,
    horizons: List[int],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "test",
    return_predictions: bool = False
) -> Dict[int, Dict[str, Union[float, torch.Tensor]]]:
    """Evaluate model on a single embedding cache dataset for all horizons.
    
    Args:
        model: Trained model
        dataset_path: Path to dataset directory in embedding cache
        horizons: List of forecast horizons
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        split: Data split to evaluate on ('test', 'val', 'train')
        return_predictions: If True, return predictions and targets
    
    Returns:
        Dictionary mapping horizon to metrics/predictions
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for horizon in horizons:
            try:
                # Load data for this dataset and horizon using embedding cache loader
                dataloader = build_embedding_cache_loader(
                    dataset_path,
                    horizon=horizon,
                    split=split,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                )
                
                all_predictions = []
                all_targets = []
                running_mse = 0.0
                running_mae = 0.0
                running_mape = 0.0
                steps = 0
                
                for embeddings, targets in tqdm(dataloader, desc=f"{split.capitalize()} H{horizon}", leave=False):
                    embeddings = embeddings.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    predictions = model(embeddings, horizon)
                    
                    # Calculate metrics
                    mse_loss = torch.nn.functional.mse_loss(predictions, targets)
                    mae_loss = torch.nn.functional.l1_loss(predictions, targets)
                    
                    # MAPE calculation (avoiding division by zero)
                    epsilon = 1e-8
                    mape_loss = torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100
                    
                    running_mse += float(mse_loss.item())
                    running_mae += float(mae_loss.item())
                    running_mape += float(mape_loss.item())
                    steps += 1
                    
                    if return_predictions:
                        all_predictions.append(predictions.cpu())
                        all_targets.append(targets.cpu())
                
                # Compile results
                result_dict = {
                    'mse': running_mse / max(1, steps),
                    'mae': running_mae / max(1, steps),
                    'mape': running_mape / max(1, steps),
                    'rmse': np.sqrt(running_mse / max(1, steps)),
                    'samples': steps * batch_size if steps > 0 else 0
                }
                
                if return_predictions and all_predictions:
                    result_dict['predictions'] = torch.cat(all_predictions, dim=0)
                    result_dict['targets'] = torch.cat(all_targets, dim=0)
                
                results[horizon] = result_dict
                
            except FileNotFoundError:
                print(f"  No {split} data for horizon {horizon}")
                results[horizon] = {
                    'mse': float('nan'), 
                    'mae': float('nan'),
                    'mape': float('nan'),
                    'rmse': float('nan'),
                    'samples': 0
                }
                if return_predictions:
                    results[horizon]['predictions'] = None
                    results[horizon]['targets'] = None
    
    return results


def evaluate_all_embedding_datasets(
    model: MultiHorizonForecastMLP,
    embedding_cache_dir: Path,
    horizons: List[int],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "test"
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Evaluate model on all embedding cache datasets.
    
    Args:
        model: Trained model
        embedding_cache_dir: Path to embedding_cache directory
        horizons: List of forecast horizons
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        split: Data split to evaluate on
    
    Returns:
        Dictionary mapping dataset_name to horizon results
    """
    datasets = discover_embedding_datasets(embedding_cache_dir)
    all_results = {}
    
    print(f"Evaluating on {len(datasets)} embedding cache datasets:")
    for group, dataset_name, dataset_path in datasets:
        print(f"  {group}/{dataset_name}")
    
    for group, dataset_name, dataset_path in datasets:
        dataset_key = f"{group}/{dataset_name}"
        print(f"\nEvaluating {dataset_key}...")
        
        dataset_results = evaluate_embedding_dataset(
            model, dataset_path, horizons, device, batch_size, num_workers, split
        )
        
        all_results[dataset_key] = dataset_results
        
        # Print results for this dataset
        print(f"Results for {dataset_key}:")
        for horizon, metrics in dataset_results.items():
            if not np.isnan(metrics['mse']):
                print(f"  H{horizon}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.6f} ({metrics['samples']} samples)")
            else:
                print(f"  H{horizon}: No data available")
    
    return all_results


def run_evaluation(config: EvalConfig) -> None:
    """Run the complete evaluation pipeline."""
    
    # Setup paths
    checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
    embedding_cache_dir = Path(config.embedding_cache_dir).expanduser().resolve()
    results_dir = Path(config.results_dir).expanduser().resolve()
    plots_dir = Path(config.plots_dir).expanduser().resolve()

    # Validate paths
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not embedding_cache_dir.exists():
        raise FileNotFoundError(f"Embedding cache directory not found: {embedding_cache_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    if config.generate_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = default_device()
    print(f"Using device: {device}")

    # Load trained model
    print(f"Loading model from: {checkpoint_path}")
    model, checkpoint_info = load_trained_model(
        checkpoint_path, device, config.mlp_hidden_dim
    )

    # Get horizons to evaluate
    if config.horizons:
        horizons = [int(h.strip()) for h in config.horizons.split(",")]
        print(f"Using custom horizons: {horizons}")
    else:
        horizons = checkpoint_info.get('horizons', [96, 192, 336, 720])
        print(f"Using model's training horizons: {horizons}")

    # Evaluate on all embedding cache datasets
    print(f"\nEvaluating on embedding cache datasets ({config.split} split)...")
    all_results = evaluate_all_embedding_datasets(
        model=model,
        embedding_cache_dir=embedding_cache_dir,
        horizons=horizons,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split=config.split
    )

    # Filter results if dataset filter is provided
    if config.dataset_filter:
        filtered_results = {
            k: v for k, v in all_results.items()
            if any(filter_name.lower() in k.lower() for filter_name in config.dataset_filter)
        }
        if not filtered_results:
            print(f"Warning: No datasets matched filter {config.dataset_filter}")
            filtered_results = all_results
        all_results = filtered_results
        print(f"Filtered to {len(all_results)} datasets: {list(all_results.keys())}")

    # Print summary
    print_evaluation_summary(all_results, checkpoint_info)

    # Save results
    json_path, csv_path = save_evaluation_results(
        all_results, checkpoint_info, results_dir, f"embedding_cache_evaluation_{config.split}"
    )

    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Generate plots if requested
    if config.generate_plots:
        print(f"\nGenerating forecast plots...")

        # Get predictions for plotting (evaluate with return_predictions=True)
        plot_results = {}
        datasets = discover_embedding_datasets(embedding_cache_dir)
        
        # Filter datasets for plotting if needed
        if config.dataset_filter:
            datasets_filtered = []
            for group, dataset_name, dataset_path in datasets:
                dataset_key = f"{group}/{dataset_name}"
                if any(filter_name.lower() in dataset_key.lower() for filter_name in config.dataset_filter):
                    datasets_filtered.append((group, dataset_name, dataset_path))
            datasets = datasets_filtered
        
        # Get the first 3 datasets for plotting
        datasets_to_plot = datasets[:3]
        
        for group, dataset_name, dataset_path in datasets_to_plot:
            dataset_key = f"{group}/{dataset_name}"
            print(f"Getting predictions for {dataset_key}...")

            dataset_predictions = evaluate_embedding_dataset(
                model=model,
                dataset_path=dataset_path,
                horizons=horizons,
                device=device,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                split=config.split,
                return_predictions=True
            )
            plot_results[dataset_key] = dataset_predictions

        # Generate plots
        if plot_results:
            figures = plot_forecast_comparison(
                results_dict=plot_results,
                datasets_to_plot=None,  # Plot all available
                horizons_to_plot=horizons[:2] if len(horizons) > 2 else horizons,  # Limit horizons for plotting
                samples_per_dataset=config.samples_per_plot,
                features_to_plot=[0],  # Plot first feature only
                save_dir=plots_dir,
                show_plots=config.show_plots
            )

            print(f"Generated {len(figures)} forecast plots in: {plots_dir}")
        else:
            print("No datasets available for plotting.")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    config = get_default_config()
    run_evaluation(config)

