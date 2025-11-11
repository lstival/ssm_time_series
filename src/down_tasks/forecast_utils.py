"""Utilities for multi-horizon forecasting evaluation and visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from dataloaders.embedding_cache_dataset import build_embedding_cache_loader


class MultiHorizonForecastMLP(nn.Module):
    """MLP that maps encoder embeddings to multiple forecast horizons.
    
    This is the updated version that handles multi-feature targets.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, horizons: List[int], target_features: int = 1) -> None:
        super().__init__()
        self.horizons = sorted(horizons)
        self.target_features = target_features
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Separate output heads for each horizon
        self.horizon_heads = nn.ModuleDict({
            str(h): nn.Linear(hidden_dim, h * target_features) for h in self.horizons
        })
    
    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        batch_size = x.shape[0]
        shared_out = self.shared_layers(x)
        output = self.horizon_heads[str(horizon)](shared_out)
        # Reshape to [batch_size, horizon, target_features]
        return output.view(batch_size, horizon, self.target_features)


def load_trained_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    mlp_hidden_dim: Optional[int] = None
) -> Tuple[MultiHorizonForecastMLP, Dict[str, Any]]:
    """Load a trained multi-horizon forecast model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        mlp_hidden_dim: Override hidden dimension if different from saved
    
    Returns:
        Tuple of (loaded_model, checkpoint_info)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    horizons = checkpoint.get('horizons', [96, 192, 336, 720])
    target_features = checkpoint.get('target_features', 1)
    
    # Try to infer dimensions from the model state dict
    model_state = checkpoint['model_state_dict']
    
    # Get input dimension from first shared layer
    input_dim = model_state['shared_layers.0.weight'].shape[1]
    
    # Get hidden dimension from first shared layer or use provided override
    if mlp_hidden_dim is not None:
        hidden_dim = mlp_hidden_dim
    else:
        hidden_dim = model_state['shared_layers.0.weight'].shape[0]
    
    print(f"Model config - Input: {input_dim}, Hidden: {hidden_dim}, Target features: {target_features}")
    print(f"Horizons: {horizons}")
    
    # Create and load model
    model = MultiHorizonForecastMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizons=horizons,
        target_features=target_features
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'avg_val_mse': checkpoint.get('avg_val_mse', 'Unknown'),
        'train_losses': checkpoint.get('train_losses', {}),
        'val_results': checkpoint.get('val_results', {}),
        'horizons': horizons,
        'target_features': target_features,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
    }
    
    print(f"Loaded model from epoch {checkpoint_info['epoch']} with avg val MSE: {checkpoint_info['avg_val_mse']}")
    
    return model, checkpoint_info


def discover_icml_datasets(icml_datasets_dir: Path) -> List[Tuple[str, Path]]:
    """Discover all ICML datasets.
    
    Args:
        icml_datasets_dir: Path to ICML_datasets directory
    
    Returns:
        List of (dataset_name, dataset_path) tuples
    """
    datasets = []
    for dataset_dir in icml_datasets_dir.iterdir():
        if dataset_dir.is_dir():
            datasets.append((dataset_dir.name, dataset_dir))
    return sorted(datasets)


def evaluate_single_dataset(
    model: MultiHorizonForecastMLP,
    dataset_path: Path,
    horizons: List[int],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "test",
    return_predictions: bool = False
) -> Dict[int, Dict[str, Union[float, torch.Tensor]]]:
    """Evaluate model on a single dataset for all horizons.
    
    Args:
        model: Trained model
        dataset_path: Path to dataset directory
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
                # Load test data for this dataset and horizon
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


def evaluate_all_icml_datasets(
    model: MultiHorizonForecastMLP,
    icml_datasets_dir: Path,
    horizons: List[int],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    split: str = "test"
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Evaluate model on all ICML datasets.
    
    Args:
        model: Trained model
        icml_datasets_dir: Path to ICML_datasets directory
        horizons: List of forecast horizons
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        split: Data split to evaluate on
    
    Returns:
        Dictionary mapping dataset_name to horizon results
    """
    datasets = discover_icml_datasets(icml_datasets_dir)
    all_results = {}
    
    print(f"Evaluating on {len(datasets)} ICML datasets:")
    for dataset_name, dataset_path in datasets:
        print(f"  {dataset_name}")
    
    for dataset_name, dataset_path in datasets:
        print(f"\nEvaluating {dataset_name}...")
        
        dataset_results = evaluate_single_dataset(
            model, dataset_path, horizons, device, batch_size, num_workers, split
        )
        
        all_results[dataset_name] = dataset_results
        
        # Print results for this dataset
        print(f"Results for {dataset_name}:")
        for horizon, metrics in dataset_results.items():
            if not np.isnan(metrics['mse']):
                print(f"  H{horizon}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, MAPE={metrics['mape']:.2f}%, RMSE={metrics['rmse']:.6f} ({metrics['samples']} samples)")
            else:
                print(f"  H{horizon}: No data available")
    
    return all_results


def save_evaluation_results(
    results: Dict[str, Dict[int, Dict[str, float]]],
    checkpoint_info: Dict[str, Any],
    output_dir: Path,
    prefix: str = "icml_evaluation"
) -> Tuple[Path, Path]:
    """Save evaluation results to JSON and CSV files.
    
    Args:
        results: Evaluation results
        checkpoint_info: Checkpoint information
        output_dir: Directory to save results
        prefix: Filename prefix
    
    Returns:
        Tuple of (json_path, csv_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'checkpoint_info': checkpoint_info,
            'results': results,
            'timestamp': timestamp
        }, f, indent=2)
    
    # Save CSV
    csv_results = []
    for dataset_name in sorted(results.keys()):
        for horizon in sorted(results[dataset_name].keys()):
            metrics = results[dataset_name][horizon]
            csv_results.append({
                'dataset_name': dataset_name,
                'horizon': horizon,
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'rmse': metrics['rmse'],
                'samples': metrics['samples'],
                'timestamp': timestamp
            })
    
    csv_df = pd.DataFrame(csv_results)
    csv_path = output_dir / f"{prefix}_{timestamp}.csv"
    csv_df.to_csv(csv_path, index=False)
    
    return json_path, csv_path


def plot_forecast_sample(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sample_idx: int = 0,
    feature_idx: int = 0,
    horizon: int = None,
    dataset_name: str = "Dataset",
    save_path: Optional[Path] = None,
    show_plot: bool = True,
    history_length: int = 96,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot a single forecast sample showing history, ground truth, and prediction.
    
    Args:
        predictions: Model predictions [batch, horizon, features]
        targets: Ground truth targets [batch, horizon, features]
        sample_idx: Index of sample to plot
        feature_idx: Index of feature to plot (for multivariate)
        horizon: Forecast horizon (for title)
        dataset_name: Name of dataset (for title)
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        history_length: Length of historical context to simulate
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if predictions is None or targets is None:
        raise ValueError("Predictions and targets cannot be None")
    
    # Convert to numpy
    pred_np = predictions[sample_idx, :, feature_idx].cpu().numpy()
    target_np = targets[sample_idx, :, feature_idx].cpu().numpy()
    
    # Create time indices
    forecast_length = len(pred_np)
    total_length = history_length + forecast_length
    
    # Simulate historical data (for visualization purposes)
    # In practice, you might want to pass actual historical data
    historical_mean = np.mean(target_np)
    historical_std = np.std(target_np)
    history = np.random.normal(historical_mean, historical_std, history_length)
    
    # Create time axis
    time_idx = np.arange(total_length)
    hist_idx = time_idx[:history_length]
    forecast_idx = time_idx[history_length:]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot historical data
    ax.plot(hist_idx, history, 'b-', label='Historical', alpha=0.7, linewidth=1)
    
    # Plot ground truth forecast
    ax.plot(forecast_idx, target_np, 'g-', label='Ground Truth', linewidth=2)
    
    # Plot predicted forecast
    ax.plot(forecast_idx, pred_np, 'r--', label='Prediction', linewidth=2)
    
    # Add vertical line to separate history from forecast
    ax.axvline(x=history_length-1, color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
    
    # Formatting
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.set_title(f'{dataset_name} - Forecast Sample {sample_idx} (H{horizon}, Feature {feature_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    mse = np.mean((pred_np - target_np) ** 2)
    mae = np.mean(np.abs(pred_np - target_np))
    
    # Add metrics text box
    textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_forecast_comparison(
    results_dict: Dict[str, Dict[int, Dict[str, Union[float, torch.Tensor]]]],
    datasets_to_plot: List[str] = None,
    horizons_to_plot: List[int] = None,
    samples_per_dataset: int = 3,
    features_to_plot: List[int] = [0],
    save_dir: Optional[Path] = None,
    show_plots: bool = True
) -> List[plt.Figure]:
    """Plot forecast comparisons for multiple datasets and horizons.
    
    Args:
        results_dict: Results dictionary with predictions and targets
        datasets_to_plot: List of dataset names to plot (None for all)
        horizons_to_plot: List of horizons to plot (None for all)
        samples_per_dataset: Number of samples to plot per dataset
        features_to_plot: List of feature indices to plot
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    
    Returns:
        List of matplotlib Figure objects
    """
    figures = []
    
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, dataset_results in results_dict.items():
        if datasets_to_plot and dataset_name not in datasets_to_plot:
            continue
            
        for horizon, horizon_results in dataset_results.items():
            if horizons_to_plot and horizon not in horizons_to_plot:
                continue
                
            predictions = horizon_results.get('predictions')
            targets = horizon_results.get('targets')
            
            if predictions is None or targets is None:
                print(f"Skipping {dataset_name} H{horizon}: No prediction data")
                continue
            
            # Plot multiple samples for this dataset/horizon
            n_samples = min(samples_per_dataset, predictions.shape[0])
            
            for sample_idx in range(n_samples):
                for feature_idx in features_to_plot:
                    if feature_idx >= predictions.shape[2]:
                        continue
                        
                    save_path = None
                    if save_dir:
                        save_path = save_dir / f"{dataset_name}_H{horizon}_sample{sample_idx}_feat{feature_idx}.png"
                    
                    fig = plot_forecast_sample(
                        predictions=predictions,
                        targets=targets,
                        sample_idx=sample_idx,
                        feature_idx=feature_idx,
                        horizon=horizon,
                        dataset_name=dataset_name,
                        save_path=save_path,
                        show_plot=show_plots
                    )
                    figures.append(fig)
    
    return figures


def print_evaluation_summary(
    results: Dict[str, Dict[int, Dict[str, float]]],
    checkpoint_info: Dict[str, Any]
) -> None:
    """Print a comprehensive evaluation summary.
    
    Args:
        results: Evaluation results
        checkpoint_info: Checkpoint information
    """
    print("\n" + "="*100)
    print("ICML DATASETS EVALUATION SUMMARY")
    print("="*100)
    print(f"Model Configuration:")
    print(f"  Input Dim: {checkpoint_info.get('input_dim', 'Unknown')}")
    print(f"  Hidden Dim: {checkpoint_info.get('hidden_dim', 'Unknown')}")
    print(f"  Target Features: {checkpoint_info.get('target_features', 'Unknown')}")
    print(f"  Horizons: {checkpoint_info.get('horizons', 'Unknown')}")
    print(f"  Training Epoch: {checkpoint_info.get('epoch', 'Unknown')}")
    print(f"  Best Val MSE: {checkpoint_info.get('avg_val_mse', 'Unknown')}")
    
    print(f"\n{'Dataset':<15} {'Horizon':<8} {'MSE':<12} {'MAE':<12} {'MAPE':<10} {'RMSE':<12} {'Samples':<8}")
    print("-"*100)
    
    # Collect all metrics for averaging
    all_mse, all_mae, all_mape, all_rmse = [], [], [], []
    
    for dataset_name in sorted(results.keys()):
        for horizon in sorted(results[dataset_name].keys()):
            metrics = results[dataset_name][horizon]
            
            mse = metrics['mse']
            mae = metrics['mae'] 
            mape = metrics['mape']
            rmse = metrics['rmse']
            samples = metrics['samples']
            
            if not np.isnan(mse):
                print(f"{dataset_name:<15} {horizon:<8} {mse:<12.6f} {mae:<12.6f} {mape:<10.2f} {rmse:<12.6f} {samples:<8}")
                all_mse.append(mse)
                all_mae.append(mae)
                all_mape.append(mape)
                all_rmse.append(rmse)
            else:
                print(f"{dataset_name:<15} {horizon:<8} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<12} {samples:<8}")
    
    # Print averages
    if all_mse:
        print("-"*100)
        avg_mse = np.mean(all_mse)
        avg_mae = np.mean(all_mae) 
        avg_mape = np.mean(all_mape)
        avg_rmse = np.mean(all_rmse)
        print(f"{'AVERAGE':<15} {'ALL':<8} {avg_mse:<12.6f} {avg_mae:<12.6f} {avg_mape:<10.2f} {avg_rmse:<12.6f} {'':<8}")
    
    print("="*100)


def safe_collate_fn(batch):
    """Custom collate function that handles tensor shape mismatches."""
    try:
        embeddings, targets = zip(*batch)
        
        # Check embedding shapes
        embedding_shapes = [e.shape for e in embeddings]
        if len(set(embedding_shapes)) > 1:
            warnings.warn(f"Embedding shape mismatch: {set(embedding_shapes)}")
            # Use only samples with the most common shape
            from collections import Counter
            most_common_shape = Counter(embedding_shapes).most_common(1)[0][0]
            filtered_batch = [(e, t) for e, t in batch if e.shape == most_common_shape]
            if filtered_batch:
                embeddings, targets = zip(*filtered_batch)
            else:
                raise RuntimeError("No valid samples in batch after filtering")
        
        # Check target shapes
        target_shapes = [t.shape for t in targets]
        if len(set(target_shapes)) > 1:
            warnings.warn(f"Target shape mismatch: {set(target_shapes)}")
            # Use only samples with the most common shape
            from collections import Counter
            most_common_shape = Counter(target_shapes).most_common(1)[0][0]
            filtered_batch = [(e, t) for e, t in batch if t.shape == most_common_shape]
            if filtered_batch:
                embeddings, targets = zip(*filtered_batch)
            else:
                raise RuntimeError("No valid samples in batch after filtering")
        
        return torch.stack(embeddings), torch.stack(targets)
    
    except Exception as e:
        print(f"Collate error: {e}")
        print(f"Batch info: {len(batch)} samples")
        for i, (emb, tgt) in enumerate(batch[:3]):  # Show first 3 samples
            print(f"  Sample {i}: embedding {emb.shape}, target {tgt.shape}")
        raise