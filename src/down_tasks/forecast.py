"""Fine-tune a lightweight forecasting head on top of a frozen encoder."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
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
import pandas as pd
from torch.optim import AdamW
import training_utils as tu
from time_series_loader import TimeSeriesDataModule

from moco_training import resolve_path
from util import (
	default_device,
	prepare_run_directory,
	load_encoder_checkpoint,
	train_epoch_dataset,
	evaluate_dataset
)

from down_tasks.forecast_utils import (
	ensure_dataloader_pred_len
)

from models.classifier import (
    ForecastRegressor,
    MultiHorizonForecastMLP
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

	try:
		horizon_values = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
	except ValueError as exc:
		raise ValueError(f"Failed to parse horizons '{args.horizons}': {exc}") from exc
	if not horizon_values:
		raise ValueError("At least one forecast horizon must be provided")
	if any(value <= 0 for value in horizon_values):
		raise ValueError(f"All horizons must be positive integers, received: {horizon_values}")
	horizons = sorted(set(horizon_values))
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

	model_cfg = dict(config.model)
	if args.token_size is not None:
		model_cfg["input_dim"] = args.token_size
	if args.model_dim is not None:
		model_cfg["model_dim"] = args.model_dim
	if args.embedding_dim is not None:
		model_cfg["embedding_dim"] = args.embedding_dim
	if args.depth is not None:
		model_cfg["depth"] = args.depth
	model_cfg.setdefault("pooling", "mean")
	model_cfg.setdefault("dropout", 0.1)
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
		train_loader = group.train
		if train_loader is None:
			print(f"Skipping dataset '{group.name}' because no train loader is available.")
			continue

		val_loader = group.val
		ensure_dataloader_pred_len(train_loader, max_horizon)
		if val_loader is not None:
			ensure_dataloader_pred_len(val_loader, max_horizon)

		try:
			sample_batch = next(iter(train_loader))
		except StopIteration:
			print(f"Skipping dataset '{group.name}' because the train loader is empty.")
			continue

		seq_x, seq_y = sample_batch[0], sample_batch[1]
		target_features = seq_y.size(2)
		available_steps = seq_y.size(1)
		if available_steps < max_horizon:
			raise ValueError(
				f"Dataset '{group.name}' provides only {available_steps} forecast steps, "
				f"but max horizon {max_horizon} was requested."
			)
		print(
			f"Dataset '{group.name}': seq_x shape {tuple(seq_x.shape)}, seq_y shape {tuple(seq_y.shape)}, "
			f"target features {target_features}"
		)

		head = MultiHorizonForecastMLP(
			input_dim=encoder.embedding_dim,
			hidden_dim=args.mlp_hidden_dim,
			horizons=horizons,
			target_features=target_features,
		).to(device)
		model = ForecastRegressor(encoder=encoder, head=head, freeze_encoder=True).to(device)
		optimizer = AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		dataset_slug = group.name.replace("\\", "__").replace("/", "__") or "dataset"
		dataset_dir = run_root / dataset_slug
		dataset_dir.mkdir(parents=True, exist_ok=True)
		best_checkpoint_path = dataset_dir / "best_model.pt"
		print(f"Training dataset '{group.name}' (artifacts -> {dataset_dir})")

		best_metric = float("inf")
		best_state: Optional[Dict[str, object]] = None
		last_train_losses: Dict[int, float] = {h: float("nan") for h in horizons}
		last_val_metrics: Optional[Dict[int, Dict[str, float]]] = None
		last_avg_train = float("nan")
		last_avg_val = float("nan")

		for epoch in range(args.epochs):
			print(f"\n[{group.name}] Epoch {epoch + 1}/{args.epochs}")
			train_losses = train_epoch_dataset(
				model,
				train_loader,
				criterion,
				optimizer,
				device,
				horizons,
			)
			val_metrics = evaluate_dataset(model, val_loader, device, horizons)

			avg_train_loss = sum(train_losses.values()) / len(train_losses)
			if val_metrics is not None:
				valid_vals = [metrics["mse"] for metrics in val_metrics.values() if not math.isnan(metrics["mse"])]
				avg_val_mse = sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")
			else:
				avg_val_mse = float("nan")

			train_str = ", ".join([f"H{h}: {train_losses[h]:.4f}" for h in horizons])
			print(f"  Train - {train_str}")
			if val_metrics is not None:
				val_parts = []
				for h in horizons:
					if h in val_metrics:
						metrics = val_metrics[h]
						val_parts.append(f"H{h}: MSE={metrics['mse']:.4f}/MAE={metrics['mae']:.4f}")
					else:
						val_parts.append(f"H{h}: MSE=nan/MAE=nan")
				val_str = ", ".join(val_parts)
				print(f"  Val   - {val_str}")
			else:
				print("  Val   - unavailable")
			print(f"  Avg Train: {avg_train_loss:.4f}, Avg Val MSE: {avg_val_mse:.4f}")

			metric_to_compare = avg_val_mse
			if math.isnan(metric_to_compare):
				metric_to_compare = avg_train_loss

			if best_state is None or metric_to_compare < best_metric:
				best_metric = metric_to_compare
				checkpoint = {
					"epoch": epoch + 1,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"train_losses": train_losses,
					"val_results": val_metrics,
					"avg_train_loss": avg_train_loss,
					"avg_val_mse": avg_val_mse,
					"horizons": horizons,
					"max_horizon": max_horizon,
					"target_features": target_features,
					"dataset": group.name,
				}
				torch.save(checkpoint, best_checkpoint_path)
				best_state = {
					"dataset": group.name,
					"dataset_slug": dataset_slug,
					"dataset_dir": str(dataset_dir),
					"train_losses": dict(train_losses),
					"avg_train_loss": avg_train_loss,
					"val_metrics": {h: dict(metrics) for h, metrics in (val_metrics or {}).items()} if val_metrics is not None else None,
					"avg_val_mse": avg_val_mse,
					"epoch": epoch + 1,
					"target_features": int(target_features),
					"checkpoint_path": str(best_checkpoint_path),
					"best_metric": metric_to_compare,
				}
				print(f"  → Saved best model (metric: {best_metric:.4f})")

			last_train_losses = dict(train_losses)
			last_val_metrics = val_metrics
			last_avg_train = avg_train_loss
			last_avg_val = avg_val_mse

		if best_state is None:
			best_state = {
				"dataset": group.name,
				"dataset_slug": dataset_slug,
				"dataset_dir": str(dataset_dir),
				"train_losses": dict(last_train_losses),
				"avg_train_loss": last_avg_train,
				"val_metrics": {h: dict(metrics) for h, metrics in (last_val_metrics or {}).items()} if last_val_metrics is not None else None,
				"avg_val_mse": last_avg_val,
				"epoch": args.epochs,
				"target_features": int(target_features),
				"checkpoint_path": str(best_checkpoint_path),
				"best_metric": best_metric,
			}
			if not best_checkpoint_path.exists():
				checkpoint = {
					"epoch": args.epochs,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"train_losses": last_train_losses,
					"val_results": last_val_metrics,
					"avg_train_loss": last_avg_train,
					"avg_val_mse": last_avg_val,
					"horizons": horizons,
					"max_horizon": max_horizon,
					"target_features": target_features,
					"dataset": group.name,
				}
				torch.save(checkpoint, best_checkpoint_path)

		best_metric_display = best_state["avg_val_mse"]
		if math.isnan(best_metric_display):
			best_metric_display = best_state["avg_train_loss"]
		print(f"\nFinished dataset '{group.name}'. Best metric: {best_metric_display:.4f}")
		print(f"Artifacts saved to: {dataset_dir}")
		dataset_records.append(best_state)

	if not dataset_records:
		print("No datasets were trained. Check dataset filters or availability.")
		return

	print("\nTraining summary:")
	for record in dataset_records:
		metric = record.get("avg_val_mse", float("nan"))
		if math.isnan(metric):
			metric = record.get("avg_train_loss", float("nan"))
		print(f"  {record['dataset']}: best epoch {record['epoch']} (metric {metric:.4f})")

	print("\n" + "=" * 80)
	print("FINAL RESULTS SUMMARY")
	print("=" * 80)
	print(f"{'Dataset':<30} {'Horizon':<8} {'TrainLoss':<12} {'ValMSE':<12} {'ValMAE':<12}")
	print("-" * 80)
	for record in dataset_records:
		train_losses = record["train_losses"]
		val_metrics = record.get("val_metrics")
		for horizon in horizons:
			train_loss = train_losses.get(horizon, float("nan"))
			if val_metrics is not None and horizon in val_metrics:
				val_mse = val_metrics[horizon]["mse"]
				val_mae = val_metrics[horizon]["mae"]
			else:
				val_mse = float("nan")
				val_mae = float("nan")
			print(f"{record['dataset']:<30} {horizon:<8} {train_loss:<12.6f} {val_mse:<12.6f} {val_mae:<12.6f}")
	print("=" * 80)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	json_path = results_dir / f"forecast_results_raw_{timestamp}.json"
	csv_path = results_dir / f"forecast_results_raw_{timestamp}.csv"

	results_payload = {
		"timestamp": timestamp,
		"checkpoint_root": str(run_root),
		"model_config": {
			"encoder_checkpoint": str(checkpoint_path),
			"embedding_dim": getattr(encoder, "embedding_dim", None),
			"hidden_dim": args.mlp_hidden_dim,
			"horizons": horizons,
			"epochs": args.epochs,
			"lr": args.lr,
			"weight_decay": args.weight_decay,
			"batch_size": batch_size,
			"val_batch_size": val_batch_size,
		},
		"datasets": dataset_records,
	}

	def _sanitize_for_json(obj: object) -> object:
		if isinstance(obj, float):
			return None if math.isnan(obj) else obj
		if isinstance(obj, dict):
			return {key: _sanitize_for_json(value) for key, value in obj.items()}
		if isinstance(obj, list):
			return [_sanitize_for_json(item) for item in obj]
		return obj

	with open(json_path, "w", encoding="utf-8") as fp:
		json.dump(_sanitize_for_json(results_payload), fp, indent=2)

	csv_rows = []
	for record in dataset_records:
		val_metrics = record.get("val_metrics")
		for horizon in horizons:
			row = {
				"dataset": record["dataset"],
				"dataset_slug": record["dataset_slug"],
				"horizon": horizon,
				"train_loss": record["train_losses"].get(horizon, float("nan")),
				"val_mse": float("nan"),
				"val_mae": float("nan"),
				"avg_train_loss": record["avg_train_loss"],
				"avg_val_mse": record.get("avg_val_mse", float("nan")),
				"best_epoch": record["epoch"],
				"target_features": record["target_features"],
				"checkpoint_path": record["checkpoint_path"],
			}
			if val_metrics is not None and horizon in val_metrics:
				row["val_mse"] = val_metrics[horizon]["mse"]
				row["val_mae"] = val_metrics[horizon]["mae"]
			csv_rows.append(row)

	pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

	print("\nResults saved to:")
	print(f"  JSON: {json_path}")
	print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
	main()
