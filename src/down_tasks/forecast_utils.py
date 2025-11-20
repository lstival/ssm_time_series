"""Utilities for multi-horizon forecasting evaluation and visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# from dataloaders.embedding_cache_dataset import build_embedding_cache_loader

from models.classifier import (
    ForecastRegressor,
    MultiHorizonForecastMLP
	)

def ensure_dataloader_pred_len(loader: Optional[DataLoader], pred_len: int) -> None:
    """Force underlying datasets to expose the requested prediction length."""

    if loader is None:
        return

    def _apply(obj: object) -> None:
        if obj is None:
            return
        if isinstance(obj, ConcatDataset):
            for child in obj.datasets:
                _apply(child)
            return
        if hasattr(obj, "dataset"):
            _apply(getattr(obj, "dataset"))
        if hasattr(obj, "pred_len"):
            setattr(obj, "pred_len", pred_len)

    _apply(getattr(loader, "dataset", None))


def compute_multi_horizon_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizons: List[int],
    criterion: nn.Module,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Compute aggregate and per-horizon losses for a batch."""

    if predictions.size(1) < targets.size(1):
        raise ValueError(
            f"Predictions shape {tuple(predictions.shape)} is shorter than targets {tuple(targets.shape)}"
        )

    loss_values: Dict[int, float] = {}
    total_loss: Optional[torch.Tensor] = None
    device = predictions.device

    for horizon in horizons:
        pred_slice = predictions[:, :horizon, :]
        target_slice = targets[:, :horizon, :]
        loss = criterion(pred_slice, target_slice)
        total_loss = loss if total_loss is None else total_loss + loss
        loss_values[horizon] = float(loss.detach().item())

    if total_loss is None:
        total_loss = torch.zeros((), device=device)
    else:
        total_loss = total_loss / max(1, len(horizons))

    return total_loss, loss_values


def compute_multi_horizon_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizons: List[int],
) -> Dict[int, Dict[str, float]]:
    """Return MSE/MAE metrics per horizon for a batch."""

    metrics: Dict[int, Dict[str, float]] = {}

    for horizon in horizons:
        pred_slice = predictions[:, :horizon, :]
        target_slice = targets[:, :horizon, :]
        mse = F.mse_loss(pred_slice, target_slice)
        mae = F.l1_loss(pred_slice, target_slice)
        metrics[horizon] = {
            "mse": float(mse.detach().item()),
            "mae": float(mae.detach().item()),
        }

    return metrics


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
    if any(key.startswith('horizon_heads.') for key in model_state.keys()):
        raise ValueError(
            "Checkpoint was created with the legacy per-horizon heads. "
            "Please retrain using the updated unified head architecture."
        )

    # Get input dimension from first shared layer
    input_dim = model_state['shared_layers.0.weight'].shape[1]

    # Get hidden dimension from first shared layer or use provided override
    if mlp_hidden_dim is not None:
        hidden_dim = mlp_hidden_dim
    else:
        hidden_dim = model_state['shared_layers.0.weight'].shape[0]

    max_horizon = max(horizons)
    head_rows = model_state['prediction_head.weight'].shape[0]
    if head_rows % max_horizon != 0:
        raise ValueError(
            "Prediction head rows are not divisible by the maximum horizon; cannot infer target features."
        )
    inferred_features = head_rows // max_horizon
    if target_features is None:
        target_features = inferred_features
    elif target_features != inferred_features:
        raise ValueError(
            "Target feature dimension mismatch between checkpoint metadata "
            f"({target_features}) and weights ({inferred_features})."
        )

    print(
        f"Model config - Input: {input_dim}, Hidden: {hidden_dim}, Target features: {target_features}, "
        f"Max horizon: {max_horizon}"
    )
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
        'max_horizon': max_horizon,
    }
    
    print(f"Loaded model from epoch {checkpoint_info['epoch']} with avg val MSE: {checkpoint_info['avg_val_mse']}")
    
    return model, checkpoint_info


def train_epoch_multi_horizon(
    model: MultiHorizonForecastMLP,
    horizon_loaders: Dict[int, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[int, float]:
    """Train on cached embeddings covering multiple horizons."""

    model.train()
    horizon_losses = {h: 0.0 for h in horizon_loaders.keys()}
    horizon_steps = {h: 0 for h in horizon_loaders.keys()}

    horizon_iters = {h: iter(loader) for h, loader in horizon_loaders.items()}
    max_steps = max(len(loader) for loader in horizon_loaders.values())

    for _ in tqdm(range(max_steps), desc="Train Multi-Horizon MLP", leave=False):
        active_horizons = 0
        optimizer.zero_grad(set_to_none=True)

        for horizon, data_iter in list(horizon_iters.items()):
            try:
                embeddings, targets = next(data_iter)
            except StopIteration:
                horizon_iters[horizon] = iter(horizon_loaders[horizon])
                continue

            embeddings = embeddings.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(embeddings, horizon)
            loss = criterion(predictions, targets)
            loss.backward()

            horizon_losses[horizon] += float(loss.detach().item())
            horizon_steps[horizon] += 1
            active_horizons += 1

        if active_horizons > 0:
            optimizer.step()

    return {h: horizon_losses[h] / max(1, horizon_steps[h]) for h in horizon_losses.keys()}


def evaluate_multi_horizon(
    model: MultiHorizonForecastMLP,
    horizon_loaders: Dict[int, Optional[DataLoader]],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[int, Optional[Dict[str, float]]]:
    """Evaluate cached-embedding dataloaders across horizons."""

    model.eval()
    results: Dict[int, Optional[Dict[str, float]]] = {}

    with torch.no_grad():
        for horizon, dataloader in horizon_loaders.items():
            if dataloader is None:
                results[horizon] = None
                continue

            running_mse = 0.0
            running_mae = 0.0
            steps = 0

            for embeddings, targets in tqdm(dataloader, desc=f"Val H{horizon}", leave=False):
                embeddings = embeddings.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                predictions = model(embeddings, horizon)
                mse_loss = criterion(predictions, targets)
                mae_loss = F.l1_loss(predictions, targets)

                running_mse += float(mse_loss.detach().item())
                running_mae += float(mae_loss.detach().item())
                steps += 1

            results[horizon] = {
                "mse": running_mse / max(1, steps),
                "mae": running_mae / max(1, steps),
            }

    return results


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


def save_evaluation_results(
    results: Dict[str, Dict[int, Dict[str, float]]],
    checkpoint_info: Dict[str, Any],
    output_dir: Path,
    prefix: str = "icml_evaluation",
    timestamp: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Save evaluation results to JSON and CSV files.
    
    Args:
        results: Evaluation results
        checkpoint_info: Checkpoint information
        output_dir: Directory to save results
        prefix: Filename prefix
        timestamp: Optional timestamp override (default uses current time)
    
    Returns:
        Tuple of (json_path, csv_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
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