"""Evaluate a trained forecasting model built on top of MambaVisualEncoder."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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


def main() -> None:
    # First parse only --config to locate the YAML schema
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        # default="configs/forecast_evaluation.yaml",
        default=r"C:\WUR\ssm_time_series\src\configs\forecast_evaluation.yaml",
        help="Path to YAML config describing CLI args",
    )
    # pre_parser.add_argument(
    #     "----checkpoint_path"
    # )
    known, _ = pre_parser.parse_known_args()
    # Normalize windows backslashes, expand ~ and env vars, then resolve to an absolute Path
    config_str = os.path.expanduser(os.path.expandvars(known.config))
    config_norm = os.path.normpath(config_str)
    config_path = Path(config_norm).resolve()

    # Load YAML schema

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        schema = yaml.safe_load(f) or {}

    # Map simple type names in YAML to Python types/callables for argparse
    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": lambda s: s.lower() in ("1", "true", "yes"),
    }

    # Build the real ArgumentParser from the schema
    parser = argparse.ArgumentParser(description="Evaluate a forecasting checkpoint (args loaded from YAML schema)")
    parser.add_argument("--config", type=str, default=str(config_path), help="Path to YAML config used to build arguments")

    for name, spec in schema.items():
        arg_name = f"--{name}"
        arg_type = type_map.get(spec.get("type", "str"), str)
        # YAML may use `null` for defaults -> Python None
        default = spec.get("default", None)
        required = bool(spec.get("required", False))
        help_text = spec.get("description", None)
        choices = spec.get("choices", None)
        # For boolean flags, use explicit type parsing rather than store_true so value can be set via config/CLI
        parser.add_argument(
            arg_name,
            type=arg_type,
            default=default,
            required=required and default is None,
            help=help_text,
            choices=choices,
        )

    # Parse final args (CLI overrides YAML defaults)
    args = parser.parse_args()

    device = default_device()
    print(f"Using device: {device}")

    data_kwargs = dict(
        data_dir=args.data_dir,
        filename=args.filename,
        dataset_name=args.dataset_name,
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
        raise RuntimeError(f"Requested split '{args.split}' is not available for evaluation.")

    sample_batch = next(iter(eval_loader))
    sample_target = sample_batch[1]
    forecast_len = sample_target.size(1) * sample_target.size(2)
    print(f"Detected forecast length: {forecast_len}")

    encoder = MambaVisualEncoder(
        input_dim=args.token_size,
        model_dim=args.model_dim,
        depth=args.depth,
        embedding_dim=args.embedding_dim,
        pooling="mean",
        dropout=0.1,
    )

    model = ForecastModel(encoder=encoder, forecast_len=forecast_len).to(device)

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_info = load_checkpoint(model, checkpoint_path, device)
    if checkpoint_info:
        epoch = checkpoint_info.get("epoch")
        if epoch is not None:
            print(f"Loaded weights from epoch {epoch}")

    metrics, predictions, targets = evaluate_model(model, eval_loader, device)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    run_dir = prepare_run_directory(Path(args.output_dir), f"forecast_eval_{args.split}")
    print(f"Saving evaluation artifacts to: {run_dir}")

    metrics_payload = {
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "metrics": metrics,
        "args": {
            "data_dir": args.data_dir,
            "filename": args.filename,
            "dataset_name": args.dataset_name,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "token_size": args.token_size,
            "model_dim": args.model_dim,
            "embedding_dim": args.embedding_dim,
            "depth": args.depth,
            "num_workers": args.num_workers,
        },
    }

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    tensor_payload = {
        "predictions": torch.from_numpy(predictions.astype(np.float32, copy=False)),
        "targets": torch.from_numpy(targets.astype(np.float32, copy=False)),
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    torch.save(tensor_payload, run_dir / "evaluation.pt")

    print("Finished evaluation.")


if __name__ == "__main__":
    main()
    #python -m evaluation.forecast_evaluation --config configs/forecast_evaluation.yaml --checkpoint_path "C:\WUR\ssm_time_series\checkpoints\forecast_encoder_20251014_2306\best.pt" --split test --output_dir "C:\WUR\ssm_time_series\eval_runs"

    # Example usage (CLI):
    # From the repository root run:
    # python -m evaluation.forecast_evaluation \
    #   --config configs/forecast_evaluation.yaml \
    #   --checkpoint_path "C:\WUR\ssm_time_series\checkpoints\forecast_encoder_20251014_2306\best.pt" \
    #   --split test \
    #   --output_dir "C:\WUR\ssm_time_series\eval_runs"
    
    # Programmatic usage (from a Python interpreter, e.g. for quick testing):
    # import sys
    # sys.argv = [
    #     "forecast_evaluation.py",
    #     "--config", "configs/forecast_evaluation.yaml",
    #     "--checkpoint_path", r"C:\WUR\ssm_time_series\checkpoints\forecast_encoder_20251014_2306\best.pt",
    #     "--split", "test",
    #     "--output_dir", r"C:\WUR\ssm_time_series\eval_runs",
    # ]
    # main()