"""Fine-tune a lightweight forecasting head on top of a frozen encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

import training_utils as tu
from moco_training import resolve_path
from util import (
	default_device,
	log_and_save,
	prepare_run_directory,
)
from time_series_loader import TimeSeriesDataModule


class ForecastMLP(nn.Module):
	"""Small MLP that maps encoder embeddings to flattened forecasts."""

	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class ForecastRegressor(nn.Module):
	"""Frozen encoder followed by a trainable forecasting head."""

	def __init__(
		self,
		*,
		encoder: nn.Module,
		head: ForecastMLP,
		freeze_encoder: bool = True,
	) -> None:
		super().__init__()
		self.encoder = encoder
		self.head = head
		self.freeze_encoder = freeze_encoder

		if freeze_encoder:
			for param in self.encoder.parameters():
				param.requires_grad = False
			self.encoder.eval()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
			embeddings = self.encoder(x)
		return self.head(embeddings)

	def train(self, mode: bool = True) -> ForecastRegressor:  # type: ignore[override]
		super().train(mode)
		if self.freeze_encoder:
			self.encoder.eval()
		return self


def load_encoder_checkpoint(
	encoder: nn.Module,
	checkpoint_path: Path,
	device: torch.device,
) -> Dict[str, object]:
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

	payload = torch.load(checkpoint_path, map_location=device)
	if not isinstance(payload, dict):
		raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")

	candidates = (
		payload.get("model_state_dict"),
		payload.get("encoder_state_dict"),
		payload.get("state_dict"),
		payload.get("encoder"),
		payload.get("model"),
	)
	state_dict = next((item for item in candidates if isinstance(item, dict)), None)
	state_dict = state_dict or payload

	missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"Warning: missing encoder weights: {sorted(missing)}")
	if unexpected:
		print(f"Warning: unexpected encoder weights: {sorted(unexpected)}")

	return payload


def train_epoch(
	model: ForecastRegressor,
	dataloader: torch.utils.data.DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
) -> float:
	model.train()
	running_loss = 0.0
	steps = 0

	for batch in tqdm(dataloader, desc="Train", leave=False):
		seq_x, seq_y, _, _ = batch
		seq_x = seq_x.to(device).float().transpose(1, 2)
		seq_y = seq_y.to(device).float()
		target = seq_y.reshape(seq_y.size(0), -1)

		optimizer.zero_grad(set_to_none=True)
		predictions = model(seq_x)
		loss = criterion(predictions, target)
		loss.backward()
		optimizer.step()

		running_loss += float(loss.item())
		steps += 1

	return running_loss / max(1, steps)


def evaluate(
	model: ForecastRegressor,
	dataloader: Optional[torch.utils.data.DataLoader],
	criterion: nn.Module,
	device: torch.device,
) -> Optional[float]:
	if dataloader is None:
		return None

	model.eval()
	running_loss = 0.0
	steps = 0

	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Val", leave=False):
			seq_x, seq_y, _, _ = batch
			seq_x = seq_x.to(device).float().transpose(1, 2)
			seq_y = seq_y.to(device).float()
			target = seq_y.reshape(seq_y.size(0), -1)

			predictions = model(seq_x)
			loss = criterion(predictions, target)

			running_loss += float(loss.item())
			steps += 1

	return running_loss / max(1, steps)


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
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-2)
	parser.add_argument("--num-workers", type=int, default=None)
	parser.add_argument("--token-size", type=int, default=None)
	parser.add_argument("--model-dim", type=int, default=None)
	parser.add_argument("--embedding-dim", type=int, default=None)
	parser.add_argument("--depth", type=int, default=None)
	parser.add_argument("--mlp-hidden-dim", type=int, default=512)
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
	run_root = prepare_run_directory(Path(checkpoint_base), "forecast_mlp")
	print(f"Checkpoint root directory: {run_root}")

	results = []
	for group in dataset_groups:
		train_loader = group.train
		if train_loader is None:
			print(f"Skipping dataset '{group.name}' because no train loader is available.")
			continue

		val_loader = group.val
		try:
			sample_batch = next(iter(train_loader))
		except StopIteration:
			print(f"Skipping dataset '{group.name}' because the train loader is empty.")
			continue

		seq_x, seq_y = sample_batch[0], sample_batch[1]
		forecast_len = seq_y.size(1) * seq_y.size(2)
		print(
			f"Dataset '{group.name}': seq_x shape {tuple(seq_x.shape)}, seq_y shape {tuple(seq_y.shape)}, "
			f"forecast length {forecast_len}"
		)

		head = ForecastMLP(encoder.embedding_dim, args.mlp_hidden_dim, forecast_len).to(device)
		model = ForecastRegressor(encoder=encoder, head=head, freeze_encoder=True).to(device)
		optimizer = AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		dataset_slug = group.name.replace("\\", "__").replace("/", "__") or "dataset"
		dataset_dir = run_root / dataset_slug
		dataset_dir.mkdir(parents=True, exist_ok=True)
		print(f"Training dataset '{group.name}' (artifacts -> {dataset_dir})")

		best_val = float("inf")
		last_train_loss = float("nan")
		for epoch in range(args.epochs):
			print(f"\n[{group.name}] Epoch {epoch + 1}/{args.epochs}")
			last_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
			val_loss = evaluate(model, val_loader, criterion, device)

			best_val = log_and_save(
				optimizer,
				models={"forecast_mlp": model},
				epoch=epoch,
				epochs=args.epochs,
				train_loss=last_train_loss,
				val_loss=val_loss,
				checkpoint_dir=dataset_dir,
				best_loss=best_val,
			)

		final_metric = best_val if val_loader is not None else last_train_loss
		print(f"\nFinished dataset '{group.name}'. Best metric: {final_metric:.4f}")
		print(f"Artifacts saved to: {dataset_dir}")
		results.append((group.name, final_metric))

	if not results:
		print("No datasets were trained. Check dataset filters or availability.")
	else:
		print("\nTraining summary:")
		for name, metric in results:
			print(f"  {name}: {metric:.4f}")


if __name__ == "__main__":
	main()
