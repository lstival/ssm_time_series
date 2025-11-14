"""Fine-tune a lightweight forecasting head on top of a frozen encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import torch.nn as nn
import training_utils as tu
from time_series_loader import TimeSeriesDataModule

from moco_training import resolve_path
from util import (
	default_device,
	prepare_run_directory,
	load_encoder_checkpoint,
)
from down_tasks.forecast_shared import (
    apply_model_overrides,
    finalize_results,
    parse_horizon_values,
    train_dataset_group,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train forecasting MLP on frozen encoder")
	parser.add_argument(
		"--data-dir",
		type=str,
		default=None,
		help="Root directory with ICML datasets",
	)
	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help="Path to model configuration YAML",
	)
	parser.add_argument(
		"--filename",
		type=str,
		default=None,
		help="Optional filename to select a specific dataset",
	)
	parser.add_argument(
		"--dataset-name",
		type=str,
		default=None,
		help="Optional dataset name filter",
	)
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--val-batch-size", type=int, default=None)
	parser.add_argument("--epochs", type=int, default=2)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-2)
	parser.add_argument("--num-workers", type=int, default=None)
	parser.add_argument("--token-size", type=int, default=None)
	parser.add_argument("--model-dim", type=int, default=None)
	parser.add_argument("--embedding-dim", type=int, default=None)
	parser.add_argument("--depth", type=int, default=None)
	parser.add_argument("--mlp-hidden-dim", type=int, default=512)
	parser.add_argument(
		"--horizons",
		type=str,
		default="96,192,336,720",
		help="Comma-separated forecast horizons",
	)
	parser.add_argument(
		"--encoder-checkpoint",
		type=str,
		default=r"C:\\WUR\\ssm_time_series\\checkpoints\\ts_encoder_20251101_1100\\time_series_best.pt",
		help="Path to pretrained encoder weights",
	)
	parser.add_argument(
		"--checkpoint-dir",
		type=str,
		default=None,
		help="Directory for forecasting checkpoints",
	)
	parser.add_argument(
		"--results-dir",
		type=str,
		default=None,
		help="Directory to save aggregated results",
	)
	parser.add_argument("--seed", type=int, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	default_cfg = SRC_DIR / "configs" / "mamba_encoder.yaml"
	config_candidate = Path(args.config) if args.config is not None else default_cfg
	config_path = resolve_path(Path.cwd(), config_candidate)
	if config_path is None or not config_path.exists():
		raise FileNotFoundError(f"Configuration file not found: {config_candidate}")
	config = tu.load_config(config_path)

	seed = args.seed if args.seed is not None else config.seed
	tu.set_seed(seed)
	torch.manual_seed(seed)

	horizons = parse_horizon_values(args.horizons)
	max_horizon = max(horizons)
	print(f"Training on horizons: {horizons} (max horizon: {max_horizon})")

	data_cfg = dict(config.data or {})
	logging_cfg = dict(config.logging or {})

	default_data_dir = data_cfg.get("data_dir")
	if default_data_dir is not None:
		data_dir_path = Path(default_data_dir)
		if not data_dir_path.is_absolute():
			default_data_dir = str((config_path.parent / data_dir_path).resolve())
		else:
			default_data_dir = str(data_dir_path)
	else:
		default_data_dir = str((ROOT_DIR / "ICML_datasets").resolve())

	data_dir = args.data_dir or default_data_dir
	data_dir_path = resolve_path(config_path.parent, data_dir)
	if data_dir_path is not None:
		data_dir = str(data_dir_path)

	filename = args.filename if args.filename is not None else data_cfg.get("filename")
	if filename is not None:
		filename_path = resolve_path(config_path.parent, filename)
		if filename_path is not None:
			filename = str(filename_path)
	dataset_name = args.dataset_name if args.dataset_name is not None else data_cfg.get("dataset_name", "")
	batch_size = args.batch_size if args.batch_size is not None else int(data_cfg.get("batch_size", 16))
	val_batch_size = args.val_batch_size if args.val_batch_size is not None else int(data_cfg.get("val_batch_size", batch_size))
	num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 4))

	checkpoint_candidate = args.checkpoint_dir or logging_cfg.get("checkpoint_dir") or (ROOT_DIR / "checkpoints")
	checkpoint_base = resolve_path(config_path.parent, checkpoint_candidate)
	if checkpoint_base is None:
		checkpoint_base = (ROOT_DIR / "checkpoints").resolve()

	device = default_device()
	print(f"Using device: {device}")

	module = TimeSeriesDataModule(
		dataset_name=dataset_name or "",
		data_dir=data_dir,
		batch_size=batch_size,
		val_batch_size=val_batch_size,
		num_workers=num_workers,
		pin_memory=True,
		normalize=True,
		filename=filename,
		train=True,
		val=True,
		test=False,
	)
	dataset_groups = module.get_dataloaders()
	if not dataset_groups:
		raise RuntimeError("No datasets available for training.")

	model_cfg = apply_model_overrides(
		config.model,
		token_size=args.token_size,
		model_dim=args.model_dim,
		embedding_dim=args.embedding_dim,
		depth=args.depth,
	)
	encoder = tu.build_encoder_from_config(model_cfg).to(device)

	checkpoint_path = resolve_path(config_path.parent, args.encoder_checkpoint)
	if checkpoint_path is None:
		checkpoint_path = Path(args.encoder_checkpoint).expanduser().resolve()
	print(f"Loading encoder checkpoint: {checkpoint_path}")
	load_encoder_checkpoint(encoder, checkpoint_path, device)

	criterion = nn.MSELoss()
	run_root = prepare_run_directory(Path(checkpoint_base), "multi_horizon_forecast_raw")
	print(f"Checkpoint root directory: {run_root}")

	results_dir = Path(args.results_dir) if args.results_dir else (ROOT_DIR / "results")
	results_dir = results_dir.expanduser().resolve()
	results_dir.mkdir(parents=True, exist_ok=True)
	print(f"Results directory: {results_dir}")

	dataset_records: List[Dict[str, object]] = []

	for group in dataset_groups:
		record = train_dataset_group(
			group_name=group.name,
			train_loader=group.train,
			val_loader=group.val,
			encoder=encoder,
			device=device,
			horizons=horizons,
			mlp_hidden_dim=args.mlp_hidden_dim,
			lr=args.lr,
			weight_decay=args.weight_decay,
			epochs=args.epochs,
			run_root=run_root,
			max_horizon=max_horizon,
			criterion=criterion,
		)
		if record is not None:
			dataset_records.append(record)

	if not dataset_records:
		print("No datasets were trained. Check dataset filters or availability.")
		return

	finalize_results(
		dataset_records=dataset_records,
		horizons=horizons,
		run_root=run_root,
		results_dir=results_dir,
		filename_prefix="forecast_results_raw",
		checkpoint_path=checkpoint_path,
		encoder_embedding_dim=getattr(encoder, "embedding_dim", None),
		mlp_hidden_dim=args.mlp_hidden_dim,
		epochs=args.epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		batch_size=batch_size,
		val_batch_size=val_batch_size,
	)


if __name__ == "__main__":
	main()
