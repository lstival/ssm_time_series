"""Evaluate dataset-specific forecasting heads on ICML time-series datasets."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from moco_training import resolve_path
from time_series_loader import TimeSeriesDataModule
from util import (
    default_device,
    load_encoder_checkpoint,
)
from down_tasks.forecast_shared import apply_model_overrides
from down_tasks.forecast_utils import (
    compute_multi_horizon_metrics,
    ensure_dataloader_pred_len,
)
from models.classifier import ForecastRegressor, MultiHorizonForecastMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen encoder + forecasting heads across ICML datasets",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration YAML (defaults to configs/mamba_encoder.yaml)",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        required=True,
        help="Directory containing per-dataset checkpoint folders with best_model.pt",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default=r"C:\\WUR\\ssm_time_series\\checkpoints\\ts_encoder_20251101_1100\\time_series_best.pt",
        help="Path to the pretrained encoder weights used during training",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root directory with ICML datasets",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory where evaluation summaries and predictions will be saved",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate (fallbacks apply when unavailable)",
    )
    parser.add_argument("--token-size", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Optional comma-separated horizons to override checkpoint metadata",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated list of checkpoint folder names to evaluate",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="forecast_eval",
        help="Prefix for generated result filenames",
    )
    return parser.parse_args()


def slug_to_relative(slug: str) -> str:
    """Recover the dataset relative path from the checkpoint folder name."""
    return slug.replace("__", "/")


def discover_dataset_checkpoints(root: Path) -> List[Tuple[str, Path]]:
    """Collect dataset checkpoint files stored under ``root``."""
    if root.is_file():
        if root.name.endswith(".pt"):
            return [(root.stem, root)]
        raise ValueError(f"Checkpoint root '{root}' must be a directory or .pt file")

    results: List[Tuple[str, Path]] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            checkpoint_path = entry / "best_model.pt"
            if checkpoint_path.exists():
                results.append((entry.name, checkpoint_path))
    if not results:
        raise FileNotFoundError(f"No best_model.pt files found under '{root}'")
    return results


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Create a CPU copy of a state dict to reuse across encoder instances."""
    return {key: tensor.detach().cpu() for key, tensor in state_dict.items()}


def infer_head_dimensions(
    model_state: Dict[str, torch.Tensor],
    max_horizon: int,
    target_features: Optional[int],
) -> Tuple[int, int, int]:
    """Infer head hidden dimension and target feature count from the checkpoint."""
    shared_key = next(
        (key for key in model_state.keys() if key.endswith("head.shared_layers.0.weight")),
        None,
    )
    if shared_key is None:
        raise KeyError("Checkpoint missing head.shared_layers.0.weight entry")
    head_weight = model_state[shared_key]
    hidden_dim = head_weight.shape[0]
    input_dim = head_weight.shape[1]

    pred_key = next(
        (key for key in model_state.keys() if key.endswith("head.prediction_head.weight")),
        None,
    )
    if pred_key is None:
        raise KeyError("Checkpoint missing head.prediction_head.weight entry")
    rows = model_state[pred_key].shape[0]
    inferred_targets = rows // max_horizon
    if target_features is None:
        target_features = inferred_targets
    elif target_features != inferred_targets:
        raise ValueError(
            "Target feature mismatch between checkpoint metadata and weights "
            f"({target_features} vs {inferred_targets})"
        )
    return hidden_dim, target_features, input_dim


def build_model(
    *,
    model_cfg: Dict[str, object],
    encoder_state: Dict[str, torch.Tensor],
    horizons: Sequence[int],
    target_features: int,
    hidden_dim: int,
    device: torch.device,
) -> ForecastRegressor:
    """Instantiate encoder + forecasting head with provided dimensions."""
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"Warning: missing encoder params when reloading base state: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected encoder params when reloading base state: {sorted(unexpected)}")

    embedding_dim = getattr(encoder, "embedding_dim", None)
    if embedding_dim is None:
        raise AttributeError("Encoder is expected to expose an 'embedding_dim' attribute")

    head = MultiHorizonForecastMLP(
        input_dim=int(embedding_dim),
        hidden_dim=int(hidden_dim),
        horizons=[int(h) for h in horizons],
        target_features=int(target_features),
    ).to(device)
    model = ForecastRegressor(encoder=encoder, head=head, freeze_encoder=True).to(device)
    return model


def evaluate_with_predictions(
    *,
    model: ForecastRegressor,
    dataloader,
    device: torch.device,
    horizons: Sequence[int],
    dataset_label: str,
) -> Tuple[Dict[int, Dict[str, float]], List[Dict[str, object]]]:
    if dataloader is None:
        raise ValueError(f"No dataloader available for dataset '{dataset_label}'")

    model.eval()
    horizons = [int(h) for h in horizons]
    running_mse = defaultdict(float)
    running_mae = defaultdict(float)
    counts = defaultdict(int)
    predictions_rows: List[Dict[str, object]] = []

    max_horizon = model.max_horizon
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {dataset_label}", leave=False):
            seq_x, seq_y, _, _ = batch
            seq_x = seq_x.to(device).float().transpose(1, 2)
            seq_y = seq_y.to(device).float()
            if seq_y.size(1) < max_horizon:
                raise ValueError(
                    f"Dataset '{dataset_label}' provides only {seq_y.size(1)} steps "
                    f"but max horizon {max_horizon} is required"
                )
            targets = seq_y[:, -max_horizon:, :]
            predictions = model(seq_x)

            batch_metrics = compute_multi_horizon_metrics(predictions, targets, horizons)
            target_features = targets.size(2)
            batch_size = targets.size(0)

            for horizon, metrics in batch_metrics.items():
                elements = batch_size * horizon * target_features
                running_mse[horizon] += metrics["mse"] * elements
                running_mae[horizon] += metrics["mae"] * elements
                counts[horizon] += elements

            preds_cpu = predictions.detach().cpu()
            targets_cpu = targets.detach().cpu()
            for sample_idx in range(batch_size):
                global_idx = sample_offset + sample_idx
                for horizon in horizons:
                    pred_slice = preds_cpu[sample_idx, :horizon, :]
                    target_slice = targets_cpu[sample_idx, :horizon, :]
                    for step_idx in range(horizon):
                        for feature_idx in range(target_features):
                            predictions_rows.append(
                                {
                                    "dataset": dataset_label,
                                    "sample_index": global_idx,
                                    "horizon": horizon,
                                    "step": step_idx + 1,
                                    "feature": feature_idx,
                                    "prediction": float(pred_slice[step_idx, feature_idx]),
                                    "target": float(target_slice[step_idx, feature_idx]),
                                }
                            )
            sample_offset += batch_size

    metrics: Dict[int, Dict[str, float]] = {}
    for horizon in horizons:
        total = counts[horizon]
        if total == 0:
            metrics[horizon] = {"mse": float("nan"), "mae": float("nan")}
        else:
            metrics[horizon] = {
                "mse": running_mse[horizon] / total,
                "mae": running_mae[horizon] / total,
            }
    return metrics, predictions_rows


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

    device = default_device()
    print(f"Using device: {device}")

    checkpoint_root_candidate = Path(args.checkpoint_root)
    checkpoint_root = resolve_path(Path.cwd(), checkpoint_root_candidate)
    if checkpoint_root is None:
        checkpoint_root = checkpoint_root_candidate.expanduser().resolve()
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root_candidate}")

    config_model_overrides = apply_model_overrides(
        config.model,
        token_size=args.token_size,
        model_dim=args.model_dim,
        embedding_dim=args.embedding_dim,
        depth=args.depth,
    )

    print("Loading base encoder weights...")
    encoder = tu.build_encoder_from_config(config_model_overrides).to(device)
    encoder_checkpoint_candidate = Path(args.encoder_checkpoint)
    encoder_checkpoint_path = resolve_path(config_path.parent, encoder_checkpoint_candidate)
    if encoder_checkpoint_path is None:
        encoder_checkpoint_path = encoder_checkpoint_candidate.expanduser().resolve()
    load_encoder_checkpoint(encoder, encoder_checkpoint_path, device)
    base_encoder_state = clone_state_dict(encoder.state_dict())
    encoder_embedding_dim = getattr(encoder, "embedding_dim", None)
    if encoder_embedding_dim is None:
        raise AttributeError("Encoder must expose 'embedding_dim'")
    del encoder

    dataset_filter: Optional[Iterable[str]] = None
    if args.datasets:
        dataset_filter = {item.strip() for item in args.datasets.split(",") if item.strip()}

    checkpoints = discover_dataset_checkpoints(checkpoint_root)
    if dataset_filter is not None:
        checkpoints = [item for item in checkpoints if item[0] in dataset_filter]
        if not checkpoints:
            raise ValueError("No matching checkpoints found for the requested datasets")

    print(f"Discovered {len(checkpoints)} dataset checkpoints under {checkpoint_root}")

    horizons_override: Optional[List[int]] = None
    if args.horizons:
        horizons_override = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
        horizons_override.sort()

    results_dir = Path(args.results_dir) if args.results_dir else (ROOT_DIR / "results")
    results_dir = results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_records: List[Dict[str, object]] = []
    all_predictions: List[Dict[str, object]] = []

    for slug, checkpoint_path in checkpoints:
        dataset_relative = slug_to_relative(slug)
        dataset_label = dataset_relative
        print(f"\nEvaluating dataset '{dataset_label}'")

        payload = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" not in payload:
            raise KeyError(f"Checkpoint {checkpoint_path} missing 'model_state_dict'")
        model_state = payload["model_state_dict"]

        checkpoint_horizons = payload.get("horizons")
        if checkpoint_horizons is None and horizons_override is None:
            raise KeyError(f"Checkpoint {checkpoint_path} missing 'horizons' metadata")
        horizons = horizons_override or [int(h) for h in checkpoint_horizons]
        horizons = sorted(set(horizons))
        max_horizon = int(payload.get("max_horizon", max(horizons)))

        target_features = payload.get("target_features")
        hidden_dim, target_features, inferred_input_dim = infer_head_dimensions(
            model_state,
            max_horizon,
            target_features,
        )
        if inferred_input_dim != int(encoder_embedding_dim):
            raise ValueError(
                f"Inferred head input dim {inferred_input_dim} does not match encoder embedding dim {encoder_embedding_dim}"
            )

        model = build_model(
            model_cfg=config_model_overrides,
            encoder_state=base_encoder_state,
            horizons=horizons,
            target_features=target_features,
            hidden_dim=hidden_dim,
            device=device,
        )
        model.load_state_dict(model_state, strict=True)

        module = TimeSeriesDataModule(
            dataset_name="",
            data_dir=data_dir,
            batch_size=args.batch_size,
            val_batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            normalize=True,
            filename=dataset_relative,
            train=False,
            val=True,
            test=True,
        )
        dataset_groups = module.get_dataloaders()
        if not dataset_groups:
            raise RuntimeError(f"No dataloaders found for dataset '{dataset_label}'")

        group = dataset_groups[0]
        loader = None
        if args.split == "test" and group.test is not None:
            loader = group.test
        elif args.split == "val" and group.val is not None:
            loader = group.val
        elif args.split == "train" and group.train is not None:
            loader = group.train
        else:
            loader = group.test or group.val or group.train
            print(
                f"  Requested split '{args.split}' unavailable; using "
                f"{'test' if group.test is not None else 'val' if group.val is not None else 'train'} split instead"
            )
        if loader is None:
            raise RuntimeError(f"Dataset '{dataset_label}' has no available dataloader")

        ensure_dataloader_pred_len(loader, max_horizon)
        metrics, prediction_rows = evaluate_with_predictions(
            model=model,
            dataloader=loader,
            device=device,
            horizons=horizons,
            dataset_label=dataset_label,
        )

        for horizon, values in metrics.items():
            metrics_records.append(
                {
                    "dataset": dataset_label,
                    "horizon": horizon,
                    "mse": values["mse"],
                    "mae": values["mae"],
                    "checkpoint": str(checkpoint_path),
                }
            )
        all_predictions.extend(prediction_rows)

        print("  Metrics:")
        for horizon in horizons:
            mse = metrics[horizon]["mse"]
            mae = metrics[horizon]["mae"]
            if math.isnan(mse) or math.isnan(mae):
                print(f"    H{horizon}: insufficient data")
            else:
                print(f"    H{horizon}: MSE={mse:.6f}, MAE={mae:.6f}")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = results_dir / f"{args.output_prefix}_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    predictions_csv = results_dir / f"{args.output_prefix}_predictions_{timestamp}.csv"
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(predictions_csv, index=False)
    else:
        predictions_csv = None

    print("\nEvaluation artifacts:")
    print(f"  Metrics CSV: {metrics_csv}")
    if predictions_csv is not None:
        print(f"  Predictions CSV: {predictions_csv}")
    else:
        print("  No predictions were recorded")


if __name__ == "__main__":
    main()
