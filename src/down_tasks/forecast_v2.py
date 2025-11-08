"""Multi-horizon forecasting using cached encoder embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
	path_str = str(path)
	if path_str not in sys.path:
		sys.path.insert(0, path_str)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd

import training_utils as tu
from util import default_device, prepare_run_directory
from dataloaders.embedding_cache_dataset import build_embedding_cache_loader


class MultiHorizonForecastMLP(nn.Module):
	"""MLP that maps encoder embeddings to multiple forecast horizons."""

	def __init__(self, input_dim: int, hidden_dim: int, horizons: List[int]) -> None:
		super().__init__()
		self.horizons = sorted(horizons)
		self.shared_layers = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
		# Separate output heads for each horizon
		self.horizon_heads = nn.ModuleDict({
			str(h): nn.Linear(hidden_dim, h) for h in self.horizons
		})

	def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
		shared_out = self.shared_layers(x)
		return self.horizon_heads[str(horizon)](shared_out)





def train_epoch_multi_horizon(
	model: MultiHorizonForecastMLP,
	horizon_loaders: Dict[int, DataLoader],
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
) -> Dict[int, float]:
	model.train()
	horizon_losses = {h: 0.0 for h in horizon_loaders.keys()}
	horizon_steps = {h: 0 for h in horizon_loaders.keys()}

	# Create iterators for all horizons
	horizon_iters = {h: iter(loader) for h, loader in horizon_loaders.items()}
	
	# Train on all horizons in a round-robin fashion
	max_steps = max(len(loader) for loader in horizon_loaders.values())
	
	# Use non_blocking transfers for better GPU utilization
	for step in tqdm(range(max_steps), desc="Train Multi-Horizon MLP", leave=False):
		total_loss = 0.0
		active_horizons = 0
		
		optimizer.zero_grad(set_to_none=True)
		
		for horizon, data_iter in horizon_iters.items():
			try:
				embeddings, targets = next(data_iter)
				embeddings = embeddings.to(device, non_blocking=True)
				targets = targets.to(device, non_blocking=True)
				
				predictions = model(embeddings, horizon)
				loss = criterion(predictions, targets)
				loss.backward()
				
				horizon_losses[horizon] += float(loss.item())
				horizon_steps[horizon] += 1
				total_loss += float(loss.item())
				active_horizons += 1
				
			except StopIteration:
				# Reset iterator when exhausted
				horizon_iters[horizon] = iter(horizon_loaders[horizon])
				continue
		
		if active_horizons > 0:
			optimizer.step()
	
	return {h: horizon_losses[h] / max(1, horizon_steps[h]) for h in horizon_losses.keys()}


def evaluate_multi_horizon(
	model: MultiHorizonForecastMLP,
	horizon_loaders: Dict[int, Optional[DataLoader]],
	criterion: nn.Module,
	device: torch.device,
) -> Dict[int, Optional[Dict[str, float]]]:
	model.eval()
	results = {}
	
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
				mae_loss = torch.nn.functional.l1_loss(predictions, targets)
				
				running_mse += float(mse_loss.item())
				running_mae += float(mae_loss.item())
				steps += 1
			
			results[horizon] = {
				'mse': running_mse / max(1, steps),
				'mae': running_mae / max(1, steps)
			}
	
	return results


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Multi-horizon forecasting with cached embeddings")
	parser.add_argument("--embedding-cache-dir", type=str, default=r"C:\WUR\ssm_time_series\embedding_cache", help="Root directory with cached embeddings")
	parser.add_argument("--config", type=str, default=None, help="Path to model configuration YAML")
	parser.add_argument("--dataset-name", type=str, default=None, help="Optional dataset name filter")
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--val-batch-size", type=int, default=None)
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes for data loading")
	parser.add_argument("--prefetch-factor", type=int, default=4, help="Number of batches loaded in advance by each worker")
	parser.add_argument("--mlp-hidden-dim", type=int, default=512)
	parser.add_argument("--horizons", type=str, default="96,192,336,720", help="Comma-separated forecast horizons")
	parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for forecasting checkpoints")
	parser.add_argument("--results-dir", type=str, default=None, help="Directory to save results")
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def discover_datasets(embedding_cache_dir: Path) -> List[Tuple[str, str, Path]]:
	"""Discover all dataset/subdataset combinations in the cache directory.
	
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


def evaluate_dataset_per_horizon(
	model: MultiHorizonForecastMLP,
	dataset_path: Path,
	horizons: List[int],
	device: torch.device,
	batch_size: int,
	num_workers: int,
) -> Dict[int, Dict[str, float]]:
	"""Evaluate model on a single dataset for all horizons."""
	model.eval()
	results = {}
	
	with torch.no_grad():
		for horizon in horizons:
			try:
				# Load test data for this dataset and horizon
				test_loader = build_embedding_cache_loader(
					dataset_path,
					horizon=horizon,
					split="test",
					batch_size=batch_size,
					shuffle=False,
					num_workers=num_workers,
					pin_memory=device.type == "cuda",
				)
				
				running_mse = 0.0
				running_mae = 0.0
				steps = 0
				
				for embeddings, targets in test_loader:
					embeddings = embeddings.to(device, non_blocking=True)
					targets = targets.to(device, non_blocking=True)
					
					predictions = model(embeddings, horizon)
					mse_loss = torch.nn.functional.mse_loss(predictions, targets)
					mae_loss = torch.nn.functional.l1_loss(predictions, targets)
					
					running_mse += float(mse_loss.item())
					running_mae += float(mae_loss.item())
					steps += 1
				
				results[horizon] = {
					'mse': running_mse / max(1, steps),
					'mae': running_mae / max(1, steps)
				}
				
			except FileNotFoundError:
				print(f"  No test data for horizon {horizon}")
				results[horizon] = {'mse': float('nan'), 'mae': float('nan')}
	
	return results


def validate_dataset_compatibility(datasets_info: List[Tuple[str, str, Path]], horizons: List[int]) -> Dict[str, Dict[int, Tuple[int, Tuple]]]:
    """Validate that all datasets have compatible embedding and target dimensions.
    
    Returns:
        Dict mapping dataset_group_key to horizon dimensions
    """
    print("Validating dataset compatibility...")
    dataset_groups = {}
    
    for group, dataset_name, dataset_path in datasets_info:
        dataset_key = f"{group}/{dataset_name}"
        
        for horizon in horizons:
            try:
                temp_loader = build_embedding_cache_loader(
                    dataset_path,
                    horizon=horizon,
                    split="train",
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
                
                sample_batch = next(iter(temp_loader))
                embedding_dim = sample_batch[0].shape[-1]
                target_shape = sample_batch[1].shape
                
                # Create a key based on embedding dim and target shape
                group_key = f"emb_{embedding_dim}_tgt_{target_shape[-1]}"
                
                if group_key not in dataset_groups:
                    dataset_groups[group_key] = {
                        'datasets': [],
                        'horizon_dims': {}
                    }
                
                if dataset_key not in dataset_groups[group_key]['datasets']:
                    dataset_groups[group_key]['datasets'].append((group, dataset_name, dataset_path))
                
                if horizon not in dataset_groups[group_key]['horizon_dims']:
                    dataset_groups[group_key]['horizon_dims'][horizon] = (embedding_dim, target_shape)
                    print(f"  Group {group_key} H{horizon}: embedding_dim={embedding_dim}, target_shape={target_shape}")
                else:
                    expected_embedding_dim, expected_target_shape = dataset_groups[group_key]['horizon_dims'][horizon]
                    if embedding_dim != expected_embedding_dim or target_shape != expected_target_shape:
                        raise ValueError(f"Inconsistent dimensions within group {group_key} for {dataset_key} H{horizon}")
                
            except FileNotFoundError:
                continue
    
    return dataset_groups


def safe_collate_fn(batch):
    """Custom collate function that handles tensor shape mismatches."""
    try:
        embeddings, targets = zip(*batch)
        
        # Check embedding shapes
        embedding_shapes = [e.shape for e in embeddings]
        if len(set(embedding_shapes)) > 1:
            print(f"Warning: Embedding shape mismatch: {set(embedding_shapes)}")
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
            print(f"Warning: Target shape mismatch: {set(target_shapes)}")
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


def main() -> None:
    args = parse_args()
    
    # Set seed
    tu.set_seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = default_device()
    print(f"Using device: {device}")
    print(f"Data loading: {args.num_workers} workers, prefetch_factor={args.prefetch_factor}")
    
    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    print(f"Training on horizons: {horizons}")
    
    # Setup directories
    embedding_cache_dir = Path(args.embedding_cache_dir).expanduser().resolve()
    if not embedding_cache_dir.exists():
        raise FileNotFoundError(f"Embedding cache directory not found: {embedding_cache_dir}")
    
    checkpoint_base = Path(args.checkpoint_dir) if args.checkpoint_dir else (ROOT_DIR / "checkpoints")
    checkpoint_base = checkpoint_base.expanduser().resolve()
    
    results_dir = Path(args.results_dir) if args.results_dir else (ROOT_DIR / "results")
    results_dir = results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all available datasets
    all_datasets = discover_datasets(embedding_cache_dir)
    if args.dataset_name:
        all_datasets = [(g, d, p) for g, d, p in all_datasets if args.dataset_name in d]
    
    if not all_datasets:
        raise RuntimeError(f"No datasets found in {embedding_cache_dir}")
    
    print(f"Found {len(all_datasets)} datasets:")
    for group, dataset, path in all_datasets:
        print(f"  {group}/{dataset}")
    
    # Validate dataset compatibility and group by dimensions
    dataset_groups = validate_dataset_compatibility(all_datasets, horizons)
    
    if not dataset_groups:
        print("No compatible datasets found.")
        return
    
    print(f"\nFound {len(dataset_groups)} compatible dataset groups:")
    for group_key, group_info in dataset_groups.items():
        print(f"  {group_key}: {len(group_info['datasets'])} datasets")
        for group, dataset, _ in group_info['datasets']:
            print(f"    {group}/{dataset}")
    
    # For now, use the largest group or let user select
    if len(dataset_groups) > 1:
        # Use the group with the most datasets
        selected_group_key = max(dataset_groups.keys(), key=lambda k: len(dataset_groups[k]['datasets']))
        print(f"\nMultiple dataset groups found. Using group: {selected_group_key}")
        print("To train on a specific group, filter datasets using --dataset-name")
    else:
        selected_group_key = next(iter(dataset_groups.keys()))
    
    selected_datasets = dataset_groups[selected_group_key]['datasets']
    horizon_dims = dataset_groups[selected_group_key]['horizon_dims']
    
    # Extract dimensions from the selected group
    embedding_dims = set(dims[0] for dims in horizon_dims.values())
    if len(embedding_dims) > 1:
        raise ValueError(f"Multiple embedding dimensions found: {embedding_dims}")
    embedding_dim = next(iter(embedding_dims))
    
    # Get target dimension from first horizon
    first_horizon_dims = next(iter(horizon_dims.values()))
    target_feature_dim = first_horizon_dims[1][-1]  # Last dimension of target shape
    
    print(f"\nUsing {len(selected_datasets)} datasets with embedding_dim={embedding_dim}, target_features={target_feature_dim}")
    
    # Collect all training data across selected datasets
    from torch.utils.data import ConcatDataset
    
    final_train_loaders = {}
    final_val_loaders = {}
    
    print("\nLoading training data...")
    for group, dataset_name, dataset_path in selected_datasets:
        print(f"Processing {group}/{dataset_name}")
        
        for horizon in horizons:
            try:
                # Load train data
                train_loader = build_embedding_cache_loader(
                    dataset_path,
                    horizon=horizon,
                    split="train",
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )
                
                if horizon not in final_train_loaders:
                    final_train_loaders[horizon] = []
                final_train_loaders[horizon].append(train_loader)
                
                # Load val data
                try:
                    val_loader = build_embedding_cache_loader(
                        dataset_path,
                        horizon=horizon,
                        split="val",
                        batch_size=args.val_batch_size or args.batch_size,
                        shuffle=False,
                        num_workers=min(args.num_workers, 4),  # Use fewer workers for validation
                        pin_memory=device.type == "cuda",
                    )
                    
                    if horizon not in final_val_loaders:
                        final_val_loaders[horizon] = []
                    final_val_loaders[horizon].append(val_loader)
                    
                except FileNotFoundError:
                    print(f"  No val split for horizon {horizon}")
                    
            except FileNotFoundError:
                print(f"  No train data for horizon {horizon}")
    
    # Combine datasets for each horizon
    combined_train_loaders = {}
    combined_val_loaders = {}
    
    for horizon in horizons:
        if horizon in final_train_loaders:
            combined_datasets = [loader.dataset for loader in final_train_loaders[horizon]]
            if combined_datasets:
                combined_dataset = ConcatDataset(combined_datasets)
                combined_train_loaders[horizon] = DataLoader(
                    combined_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                    prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                    persistent_workers=args.num_workers > 0,
                    collate_fn=safe_collate_fn,  # Use custom collate function
                )
                print(f"Train horizon {horizon}: {len(combined_dataset)} samples")
        
        if horizon in final_val_loaders:
            combined_val_datasets = [loader.dataset for loader in final_val_loaders[horizon]]
            if combined_val_datasets:
                combined_val_dataset = ConcatDataset(combined_val_datasets)
                combined_val_loaders[horizon] = DataLoader(
                    combined_val_dataset,
                    batch_size=args.val_batch_size or args.batch_size,
                    shuffle=False,
                    num_workers=min(args.num_workers, 4),  # Use fewer workers for validation
                    pin_memory=device.type == "cuda",
                    prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                    persistent_workers=args.num_workers > 0,
                    collate_fn=safe_collate_fn,  # Use custom collate function
                )
                print(f"Val horizon {horizon}: {len(combined_val_dataset)} samples")
        else:
            combined_val_loaders[horizon] = None
    
    if not combined_train_loaders:
        raise RuntimeError("No training data available for any horizon")
    
    # Update model to handle multi-feature targets
    class MultiHorizonForecastMLP(nn.Module):
        """MLP that maps encoder embeddings to multiple forecast horizons."""
        
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
    
    # Create model with correct target dimension
    model = MultiHorizonForecastMLP(
        input_dim=embedding_dim,
        hidden_dim=args.mlp_hidden_dim,
        horizons=horizons,
        target_features=target_feature_dim,
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create checkpoint directory
    run_root = prepare_run_directory(checkpoint_base, f"multi_horizon_forecast_{selected_group_key}")
    print(f"Checkpoints will be saved to: {run_root}")
    
    # Training loop
    best_avg_val = float("inf")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_losses = train_epoch_multi_horizon(
            model, combined_train_loaders, criterion, optimizer, device
        )
        avg_train_loss = sum(train_losses.values()) / len(train_losses)
        
        # Validate
        val_results = evaluate_multi_horizon(
            model, combined_val_loaders, criterion, device
        )
        valid_val_results = {h: res for h, res in val_results.items() if res is not None}
        avg_val_mse = sum(res['mse'] for res in valid_val_results.values()) / len(valid_val_results) if valid_val_results else float("nan")
        
        # Print results
        train_str = ", ".join([f"H{h}: {loss:.4f}" for h, loss in train_losses.items()])
        val_str = ", ".join([f"H{h}: MSE={res['mse']:.4f}/MAE={res['mae']:.4f}" for h, res in valid_val_results.items()])
        print(f"  Train - {train_str}")
        print(f"  Val   - {val_str}")
        print(f"  Avg Train: {avg_train_loss:.4f}, Avg Val MSE: {avg_val_mse:.4f}")
        
        # Save checkpoint
        if not torch.isnan(torch.tensor(avg_val_mse)) and avg_val_mse < best_avg_val:
            best_avg_val = avg_val_mse
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_results": val_results,
                "avg_val_mse": avg_val_mse,
                "horizons": horizons,
                "target_features": target_feature_dim,
            }
            torch.save(checkpoint, run_root / "best_model.pt")
            print(f"  → Saved best model (avg val MSE: {best_avg_val:.4f})")
    
    print(f"\nTraining completed. Best average validation MSE: {best_avg_val:.4f}")
    print(f"Model saved to: {run_root / 'best_model.pt'}")
    
    # Evaluate on test sets per dataset
    print("\nEvaluating on test sets...")
    all_results = {}
    
    for group, dataset_name, dataset_path in selected_datasets:
        dataset_key = f"{group}/{dataset_name}"
        print(f"Evaluating {dataset_key}")
        
        dataset_results = evaluate_dataset_per_horizon(
            model, dataset_path, horizons, device, 
            args.val_batch_size or args.batch_size, min(args.num_workers, 2)  # Use fewer workers for evaluation
        )
        
        all_results[dataset_key] = dataset_results
        
        # Print results for this dataset
        for horizon, metrics in dataset_results.items():
            print(f"  H{horizon}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"forecast_results_{selected_group_key}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model_config': {
                'embedding_dim': embedding_dim,
                'hidden_dim': args.mlp_hidden_dim,
                'target_features': target_feature_dim,
                'horizons': horizons,
                'epochs': args.epochs,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'dataset_group': selected_group_key,
            },
            'results': all_results,
            'training_info': {
                'best_val_mse': best_avg_val,
                'checkpoint_path': str(run_root / "best_model.pt"),
            }
        }, f, indent=2)  # Added 'f' as the second argument
    
    # Save results as CSV
    csv_results = []
    for dataset_key in sorted(all_results.keys()):
        dataset_group, dataset_name = dataset_key.split('/', 1)
        for horizon in sorted(horizons):
            metrics = all_results[dataset_key][horizon]
            csv_results.append({
                'dataset_group': dataset_group,
                'dataset_name': dataset_name,
                'dataset_key': dataset_key,
                'horizon': horizon,
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'embedding_dim': embedding_dim,
                'hidden_dim': args.mlp_hidden_dim,
                'target_features': target_feature_dim,
                'epochs': args.epochs,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'dataset_group_key': selected_group_key,
                'best_val_mse': best_avg_val,
                'timestamp': timestamp
            })
    
    csv_df = pd.DataFrame(csv_results)
    csv_file = results_dir / f"forecast_results_{selected_group_key}_{timestamp}.csv"
    csv_df.to_csv(csv_file, index=False)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {results_file}")
    print(f"  CSV:  {csv_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<30} {'Horizon':<8} {'MSE':<12} {'MAE':<12}")
    print("-"*80)
    
    for dataset_key in sorted(all_results.keys()):
        for horizon in sorted(horizons):
            metrics = all_results[dataset_key][horizon]
            print(f"{dataset_key:<30} {horizon:<8} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f}")
    
    print("="*80)


if __name__ == "__main__":
	main()
