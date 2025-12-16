"""Fine-tune a forecasting head on top of a frozen encoder using Chronos datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import comet_ml
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

import sys

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from util import default_device, prepare_run_directory, load_encoder_checkpoint
from down_tasks.forecast_shared import (
    apply_model_overrides,
    finalize_results,
    train_dataset_group,
)
from dataloaders.utils import (
    ChronosDatasetGroup,
    build_dataset_group,
    _build_aggregated_loader,
)
from models.utils import load_chronos_forecast_config


if __name__ == "__main__":
    # Initialize Comet ML experiment from config
    from comet_utils import create_comet_experiment
    experiment = create_comet_experiment("forecast_chronos_temporal")
    
    config = load_chronos_forecast_config()

    print(f"Training on horizons: {config.horizons} (max horizon: {config.max_horizon})")
    print(f"Gradient clipping max norm: {config.grad_clip}")
    
    # Log configuration to Comet
    experiment.log_parameters({
        "horizons": config.horizons,
        "max_horizon": config.max_horizon,
        "grad_clip": config.grad_clip,
        "seed": config.seed,
    })

    tu.set_seed(config.seed)
    torch.manual_seed(config.seed)

    base_config = tu.load_config(config.model_config_path)

    print(f"Using offline cache directory: {config.load_kwargs['offline_cache_dir']}")
    if config.load_kwargs.get("force_offline"):
        print("Forcing offline mode - no network access will be attempted")

    device = default_device()
    print(f"Using device: {device}")

    model_cfg = apply_model_overrides(
        base_config.model,
        token_size=config.overrides.get("token_size"),
        model_dim=config.overrides.get("model_dim"),
        embedding_dim=config.overrides.get("embedding_dim"),
        depth=config.overrides.get("depth"),
    )
    encoder = tu.build_encoder_from_config(model_cfg).to(device)

    if config.context_length < getattr(encoder, "input_dim", 1):
        raise ValueError(
            f"context_length {config.context_length} must be >= encoder token size {getattr(encoder, 'input_dim', '?')}."
        )

    print(f"Loading encoder checkpoint: {config.encoder_checkpoint_path}")
    load_encoder_checkpoint(encoder, config.encoder_checkpoint_path, device)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Log additional configuration to Comet
    experiment.log_parameters({
        "context_length": config.context_length,
        "stride": config.stride,
        "split": config.split,
        "batch_size": config.batch_size,
        "val_batch_size": config.val_batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "mlp_hidden_dim": config.mlp_hidden_dim,
        "encoder_checkpoint": str(config.encoder_checkpoint_path),
    })

    criterion = nn.MSELoss()
    run_root = prepare_run_directory(config.checkpoint_dir, "multi_horizon_forecast_chronos")
    print(f"Checkpoint root directory: {run_root}")
    print(f"Results directory: {config.results_dir}")

    torch_dtype = torch.float32

    dataset_groups: List[ChronosDatasetGroup] = []
    for dataset_name in config.datasets_to_load:
        group = build_dataset_group(
            dataset_name,
            repo_id=config.repo_id,
            split=config.split,
            target_dtype=config.target_dtype,
            normalize_per_series=config.normalize_per_series,
            load_kwargs=config.load_kwargs,
            context_length=config.context_length,
            horizon=config.max_horizon,
            stride=config.stride,
            batch_size=config.batch_size,
            val_batch_size=config.val_batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            torch_dtype=torch_dtype,
            max_windows_per_series=config.max_windows_per_series,
            max_series=config.max_series,
            val_ratio=config.val_ratio,
            seed=config.seed,
        )
        if group is not None:
            dataset_groups.append(group)

    if not dataset_groups:
        print("No datasets were prepared. Check Chronos configuration or dataset availability.")

    dataset_records: List[Dict[str, object]] = []

    combined_train_loader = (
        _build_aggregated_loader(
            dataset_groups,
            is_train=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        if dataset_groups
        else None
    )
    combined_val_loader = (
        _build_aggregated_loader(
            dataset_groups,
            is_train=False,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        if dataset_groups
        else None
    )

    if combined_train_loader is None:
        print("No training data available after aggregating all datasets.")
    else:
        combined_metadata = {
            "datasets": [group.name for group in dataset_groups],
            "train_windows": sum(group.metadata.get("train_windows", 0) for group in dataset_groups),
            "val_windows": sum(group.metadata.get("val_windows", 0) for group in dataset_groups),
        }
        print(
            "Training shared model on aggregated datasets "
            f"{combined_metadata['datasets']} -> {combined_metadata['train_windows']} windows."
        )
        record = train_dataset_group(
            group_name="all_datasets",
            train_loader=combined_train_loader,
            val_loader=combined_val_loader,
            encoder=encoder,
            device=device,
            horizons=config.horizons,
            mlp_hidden_dim=config.mlp_hidden_dim,
            lr=config.lr,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            run_root=run_root,
            max_horizon=config.max_horizon,
            criterion=criterion,
            experiment=experiment,
            reverse_normalization=False,
        )
        if record is not None:
            dataset_records.append(record)
    
    # End Comet experiment
    experiment.end()

    if not dataset_records:
        print("No datasets were trained. Check dataset filters or availability.")

    finalize_results(
        dataset_records=dataset_records,
        horizons=config.horizons,
        run_root=run_root,
        results_dir=config.results_dir,
        filename_prefix="forecast_results_chronos",
        checkpoint_path=config.encoder_checkpoint_path,
        encoder_embedding_dim=getattr(encoder, "embedding_dim", None),
        mlp_hidden_dim=config.mlp_hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        extra_model_config={
            "context_length": config.context_length,
            "stride": config.stride,
            "split": config.split,
            "grad_clip": config.grad_clip,
        },
    )
