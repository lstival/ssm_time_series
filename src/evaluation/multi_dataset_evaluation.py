"""Evaluate a forecasting checkpoint across multiple datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Ensure sibling packages under src are importable when running as a script from subdirectories
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_provider.utils.metrics import metric as compute_metric_values
from models.mamba_visual_encoder import MambaVisualEncoder
from util import build_time_series_dataloaders, default_device, prepare_run_directory


DEFAULT_DATASETS: Dict[str, Dict[str, str]] = {
    "ETTm1": {"relative_dir": "ICML_datasets/ETT-small", "filename": "ETTm1.csv"},
    "ETTm2": {"relative_dir": "ICML_datasets/ETT-small", "filename": "ETTm2.csv"},
    "ETTh1": {"relative_dir": "ICML_datasets/ETT-small", "filename": "ETTh1.csv"},
    "ETTh2": {"relative_dir": "ICML_datasets/ETT-small", "filename": "ETTh2.csv"},
    "Traffic": {"relative_dir": "ICML_datasets/traffic", "filename": "traffic.csv"},
    "Weather": {"relative_dir": "ICML_datasets/weather", "filename": "weather.csv"},
    "Exchange": {"relative_dir": "ICML_datasets/exchange_rate", "filename": "exchange_rate.csv"},
    "Solar": {"relative_dir": "ICML_datasets/Solar", "filename": "solar_AL.csv"},
    "Electricity": {"relative_dir": "ICML_datasets/electricity", "filename": "electricity.csv"},
}


DISCOVERY_PATTERNS: Dict[str, List[str]] = {
    "ETTm1": ["ETTm1.csv", "ettm1.csv"],
    "ETTm2": ["ETTm2.csv", "ettm2.csv"],
    "ETTh1": ["ETTh1.csv", "etth1.csv"],
    "ETTh2": ["ETTh2.csv", "etth2.csv"],
    "Traffic": ["traffic.csv", "Traffic.csv"],
    "Weather": ["weather.csv", "Weather.csv"],
    "Exchange": ["exchange_rate.csv", "Exchange.csv"],
    "Solar": ["solar_AL.csv", "solar.csv", "Solar.csv"],
    "Electricity": ["electricity.csv", "Electricity.csv"],
}


class ForecastModel(nn.Module):
    def __init__(self, encoder: MambaVisualEncoder, forecast_len: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.forecast_head = nn.Linear(encoder.embedding_dim, forecast_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return self.forecast_head(embeddings)


def compute_metrics(predictions: List[np.ndarray], targets: List[np.ndarray]) -> Dict[str, float]:
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
    model.eval()
    criterion = nn.MSELoss()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            seq_x, seq_y, *_ = batch
            seq_x = seq_x.to(device).float().transpose(1, 2)
            seq_y = seq_y.to(device).float()
            target = seq_y.reshape(seq_y.size(0), -1)

            pred = model(seq_x)
            loss = criterion(pred, target)

            losses.append(loss.item())
            preds.append(pred.detach().cpu().numpy())
            trues.append(target.detach().cpu().numpy())

    metrics = compute_metrics(preds, trues)
    if losses:
        metrics["mse_loss"] = float(np.mean(losses))

    preds_arr = np.concatenate(preds, axis=0) if preds else np.empty((0, 0), dtype=np.float32)
    trues_arr = np.concatenate(trues, axis=0) if trues else np.empty((0, 0), dtype=np.float32)
    return metrics, preds_arr, trues_arr


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def load_schema_defaults(config_path: Path) -> Tuple[Dict[str, object], List[str], Dict[str, str]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle) or {}

    defaults: Dict[str, object] = {}
    required: List[str] = []
    types: Dict[str, str] = {}

    for name, spec in schema.items():
        defaults[name] = spec.get("default")
        types[name] = spec.get("type", "str")
        if spec.get("required", False) and spec.get("default") is None:
            required.append(name)

    return defaults, required, types


def cast_from_type(value: object, type_name: str) -> object:
    if value is None:
        return None

    converters = {
        "str": str,
        "int": int,
        "float": float,
        "bool": lambda x: x if isinstance(x, bool) else str(x).lower() in {"1", "true", "yes", "y"},
    }

    convert = converters.get(type_name, str)
    try:
        return convert(value)
    except Exception:
        return value


def apply_config_defaults(args: argparse.Namespace, defaults: Dict[str, object], types: Dict[str, str], required: List[str]) -> None:
    for key, default in defaults.items():
        if getattr(args, key, None) is None and default is not None:
            setattr(args, key, cast_from_type(default, types.get(key, "str")))

    missing = [key for key in required if getattr(args, key, None) is None]
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")


def discover_datasets(base_dir: Path) -> Dict[str, Dict[str, Path]]:
    results: Dict[str, Dict[str, Path]] = {}
    for name, patterns in DISCOVERY_PATTERNS.items():
        for pattern in patterns:
            matches = list(base_dir.rglob(pattern))
            if matches:
                found = matches[0]
                results[name] = {"data_dir": found.parent, "filename": found.name}
                break
    return results


def build_dataset_map(base_dir: Path, auto_discover: bool) -> Dict[str, Dict[str, Path]]:
    if auto_discover:
        return discover_datasets(base_dir)

    mapping: Dict[str, Dict[str, Path]] = {}
    for name, spec in DEFAULT_DATASETS.items():
        data_dir = base_dir / spec["relative_dir"]
        mapping[name] = {"data_dir": data_dir, "filename": spec["filename"]}
    return mapping


def prepare_model(args: argparse.Namespace, forecast_len: int, device: torch.device) -> ForecastModel:
    encoder = MambaVisualEncoder(
        input_dim=args.token_size,
        model_dim=args.model_dim,
        depth=args.depth,
        embedding_dim=args.embedding_dim,
        pooling="mean",
        dropout=0.1,
    )
    return ForecastModel(encoder, forecast_len).to(device)


def evaluate_dataset(
    name: str,
    spec: Dict[str, Path],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, float]:
    data_dir = Path(spec["data_dir"]).expanduser().resolve()
    filename = spec["filename"]
    data_file = data_dir / filename

    print(f"\nEvaluating {name} -> {data_file}")

    if not data_file.exists():
        message = f"Missing dataset file: {data_file}"
        print(f"  {message}")
        return {"mae": float("nan"), "mse": float("nan"), "error": message}

    loaders = build_time_series_dataloaders(
        data_dir=str(data_dir),
        filename=filename,
        dataset_name=filename,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        train=True,
        val=True,
        test=args.split == "test",
    )

    if args.split == "test":
        train_loader, val_loader, test_loader = loaders
        eval_loader = test_loader
    else:
        train_loader, val_loader = loaders
        eval_loader = train_loader if args.split == "train" else val_loader

    if eval_loader is None:
        print(f"  No {args.split} split available")
        return {"mae": float("nan"), "mse": float("nan"), "error": "Missing split"}

    sample_batch = next(iter(eval_loader))
    forecast_len = sample_batch[1].size(1) * sample_batch[1].size(2)

    model = prepare_model(args, forecast_len, device)
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    print(f"  Loading checkpoint {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)

    metrics, predictions, targets = evaluate_model(model, eval_loader, device)

    print(
        f"  Done. MSE={metrics.get('mse', float('nan')):.3f} "
        f"MAE={metrics.get('mae', float('nan')):.3f} "
        f"Samples={metrics.get('num_samples', 0)}"
    )

    dataset_dir = run_dir / name.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": name,
        "data_file": str(data_file),
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "metrics": metrics,
    }

    with (dataset_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    tensors = {
        "predictions": torch.from_numpy(predictions.astype(np.float32, copy=False)),
        "targets": torch.from_numpy(targets.astype(np.float32, copy=False)),
        "metrics": metrics,
        "dataset": name,
    }
    torch.save(tensors, dataset_dir / "evaluation.pt")

    return metrics


def create_results_table(results: Dict[str, Dict[str, float]], model_name: str) -> pd.DataFrame:
    ordered = list(DEFAULT_DATASETS.keys())
    table = {"Dataset": ordered, f"{model_name}_MSE": [], f"{model_name}_MAE": []}

    for dataset in ordered:
        metrics = results.get(dataset, {})
        mse = metrics.get("mse")
        mae = metrics.get("mae")
        table[f"{model_name}_MSE"].append(f"{mse:.3f}" if mse is not None and not np.isnan(mse) else "N/A")
        table[f"{model_name}_MAE"].append(f"{mae:.3f}" if mae is not None and not np.isnan(mae) else "N/A")

    return pd.DataFrame(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a forecasting checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="C:/WUR/ssm_time_series/src/configs/forecast_evaluation.yaml",
        help="YAML config file with evaluation defaults",
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--token_size", type=int, default=None)
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--base_data_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="MambaEncoder")
    parser.add_argument("--auto_discover", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    defaults, required, types = load_schema_defaults(config_path)
    apply_config_defaults(args, defaults, types, required)

    if args.base_data_dir is None:
        args.base_data_dir = defaults.get("data_dir") or "C:/WUR/ssm_time_series"

    if args.output_dir is None:
        args.output_dir = defaults.get("output_dir") or "./eval_runs"

    device = default_device()
    print(f"Using device: {device}")

    run_dir = prepare_run_directory(Path(args.output_dir), f"multi_dataset_eval_{args.split}")
    print(f"Saving outputs to {run_dir}")

    dataset_map = build_dataset_map(Path(args.base_data_dir), args.auto_discover)
    if not dataset_map:
        print("No datasets found. Check --base_data_dir or enable --auto_discover.")
        return

    results: Dict[str, Dict[str, float]] = {}
    for name, spec in dataset_map.items():
        results[name] = evaluate_dataset(name, spec, args, device, run_dir)

    results_table = create_results_table(results, args.model_name)
    print("\nSummary table:\n", results_table.to_string(index=False))

    results_table.to_csv(run_dir / "results_table.csv", index=False)
    results_table.to_excel(run_dir / "results_table.xlsx", index=False)

    summary = {
        "model_name": args.model_name,
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "evaluation_timestamp": pd.Timestamp.now().isoformat(),
        "results_by_dataset": results,
        "summary_stats": {
            "avg_mse": float(np.nanmean([metrics.get("mse", np.nan) for metrics in results.values()])),
            "avg_mae": float(np.nanmean([metrics.get("mae", np.nan) for metrics in results.values()])),
            "datasets_evaluated": int(
                sum(1 for metrics in results.values() if not np.isnan(metrics.get("mse", np.nan)))
            ),
            "total_datasets": len(results),
        },
    }

    with (run_dir / "complete_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nArtifacts written:")
    print(f"  {run_dir / 'results_table.csv'}")
    print(f"  {run_dir / 'results_table.xlsx'}")
    print(f"  {run_dir / 'complete_results.json'}")


if __name__ == "__main__":
    main()