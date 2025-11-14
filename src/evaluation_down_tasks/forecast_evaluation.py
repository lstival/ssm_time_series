"""Evaluate trained forecasting heads on ICML datasets."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from down_tasks.forecast_shared import apply_model_overrides, parse_horizon_values
from down_tasks.forecast_utils import compute_multi_horizon_metrics, ensure_dataloader_pred_len
from models.classifier import ForecastRegressor, MultiHorizonForecastMLP
from time_series_loader import TimeSeriesDataModule
from util import default_device
from moco_training import resolve_path
from dataloaders.concat_loader import DatasetLoaders


@dataclass
class CheckpointEntry:
    source: str
    path: Path
    dataset_name: str
    horizons: List[int]
    checkpoint: Dict[str, object]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forecasting checkpoints on ICML datasets")
    parser.add_argument("--config", type=str, default=None, help="Path to model configuration YAML")
    parser.add_argument("--forecast-paths", type=str, nargs="*", default=None, help="Paths to forecast run dirs or checkpoint files")
    parser.add_argument("--chronos-paths", type=str, nargs="*", default=None, help="Paths to Chronos run dirs or checkpoint files")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory with ICML datasets")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"), help="Dataset split preference")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--token-size", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--results-dir", type=str, default=None, help="Directory to save evaluation CSV")
    parser.add_argument("--output-name", type=str, default=None, help="Optional filename prefix for results")
    parser.add_argument("--device", type=str, default=None, help="Override evaluation device")
    parser.add_argument("--horizons", type=str, default=None, help="Optional comma separated horizons to enforce")
    return parser.parse_args()


def _collect_paths(raw: Optional[Sequence[str]]) -> List[Path]:
    if not raw:
        return []
    collected: List[Path] = []
    for entry in raw:
        path = Path(entry).expanduser().resolve()
        if path.is_file():
            collected.append(path)
        elif path.is_dir():
            collected.extend(sorted(path.rglob("best_model.pt")))
        else:
            print(f"Warning: path not found, skipping: {entry}")
    return collected


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    return torch.load(path, map_location=device)


def _prepare_checkpoint_entries(
    *,
    paths: Sequence[Path],
    source: str,
    enforced_horizons: Optional[List[int]],
    device: torch.device,
) -> List[CheckpointEntry]:
    entries: List[CheckpointEntry] = []
    for path in paths:
        try:
            payload = _load_checkpoint(path, device)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: failed to load checkpoint {path}: {exc}")
            continue
        dataset_name = str(payload.get("dataset") or path.parent.name)
        horizons = list(payload.get("horizons", []))
        if not horizons and enforced_horizons:
            horizons = list(enforced_horizons)
        if enforced_horizons and sorted(set(horizons)) != sorted(set(enforced_horizons)):
            print(
                f"Warning: checkpoint {path} horizons {horizons} do not match requested {enforced_horizons}; using intersection"
            )
            shared = sorted(set(horizons).intersection(enforced_horizons))
            if not shared:
                print(f"  Skipping {path} because horizons do not overlap")
                continue
            horizons = shared
        if not horizons:
            print(f"Warning: checkpoint {path} missing horizons; skipping")
            continue
        entries.append(CheckpointEntry(source=source, path=path, dataset_name=dataset_name, horizons=sorted(horizons), checkpoint=payload))
    return entries


def _select_loader(group, preferred: str):
    order = []
    if preferred == "test":
        order = [("test", group.test), ("val", group.val), ("train", group.train)]
    elif preferred == "val":
        order = [("val", group.val), ("test", group.test), ("train", group.train)]
    else:
        order = [("train", group.train), ("val", group.val), ("test", group.test)]
    for name, loader in order:
        if loader is not None:
            return loader, name
    return None, preferred


def _find_dataset_group(dataset_map: Dict[str, DatasetLoaders], entry: CheckpointEntry) -> Optional[DatasetLoaders]:
    group = dataset_map.get(entry.dataset_name)
    if group is not None:
        return group
    slug = entry.path.parent.name
    for name, candidate in dataset_map.items():
        candidate_slug = name.replace("\\", "__").replace("/", "__")
        if candidate_slug == slug:
            return candidate
    return None


def _build_model(
    *,
    checkpoint: Dict[str, object],
    horizons: Sequence[int],
    model_cfg: Dict[str, object],
    device: torch.device,
) -> ForecastRegressor:
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing model_state_dict")
    head_weight = state_dict.get("head.shared_layers.0.weight")
    if head_weight is None:
        raise ValueError("Checkpoint missing head weights")
    input_dim = head_weight.shape[1]
    hidden_dim = head_weight.shape[0]
    target_features = int(checkpoint.get("target_features", 1))
    head = MultiHorizonForecastMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        horizons=list(horizons),
        target_features=target_features,
    ).to(device)
    model = ForecastRegressor(encoder=encoder, head=head, freeze_encoder=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _evaluate_loader(
    *,
    model: ForecastRegressor,
    loader,
    horizons: Sequence[int],
    device: torch.device,
    dataset_name: str,
    split_name: str,
    checkpoint_id: str,
    source: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    max_horizon = max(horizons)
    ensure_dataloader_pred_len(loader, max_horizon)
    aggregate_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    total_metrics: Dict[int, Dict[str, float]] = {h: {"mse": 0.0, "mae": 0.0, "count": 0.0} for h in horizons}
    sample_index = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{dataset_name} [{checkpoint_id}]", leave=False):
            seq_x, seq_y, _, _ = batch
            seq_x = seq_x.to(device).float().transpose(1, 2)
            seq_y = seq_y.to(device).float()
            if seq_y.size(1) < max_horizon:
                raise ValueError(
                    f"Target sequence length {seq_y.size(1)} smaller than required max horizon {max_horizon}"
                )
            targets = seq_y[:, -max_horizon:, :]
            predictions = model(seq_x)
            batch_metrics = compute_multi_horizon_metrics(predictions, targets, list(horizons))
            batch_size = seq_x.size(0)
            target_features = targets.size(2)
            for horizon, metrics in batch_metrics.items():
                elements = float(batch_size * horizon * target_features)
                total_metrics[horizon]["mse"] += metrics["mse"] * elements
                total_metrics[horizon]["mae"] += metrics["mae"] * elements
                total_metrics[horizon]["count"] += elements
            for offset in range(batch_size):
                sample_targets = targets[offset]
                sample_preds = predictions[offset]
                for horizon in horizons:
                    target_slice = sample_targets[:horizon]
                    pred_slice = sample_preds[:horizon]
                    mse = float(F.mse_loss(pred_slice, target_slice).item())
                    mae = float(F.l1_loss(pred_slice, target_slice).item())
                    sample_rows.append(
                        {
                            "checkpoint": checkpoint_id,
                            "source": source,
                            "dataset": dataset_name,
                            "split": split_name,
                            "sample_index": sample_index + offset,
                            "horizon": horizon,
                            "mse": mse,
                            "mae": mae,
                        }
                    )
            sample_index += batch_size

    for horizon in horizons:
        count = total_metrics[horizon]["count"]
        if count == 0:
            continue
        aggregate_rows.append(
            {
                "checkpoint": checkpoint_id,
                "source": source,
                "dataset": dataset_name,
                "split": split_name,
                "horizon": horizon,
                "mse": total_metrics[horizon]["mse"] / count,
                "mae": total_metrics[horizon]["mae"] / count,
            }
        )
    return aggregate_rows, sample_rows


def main() -> None:
    args = _parse_args()

    default_config = SRC_DIR / "configs" / "mamba_encoder.yaml"
    config_candidate = Path(args.config) if args.config is not None else default_config
    config_path = resolve_path(Path.cwd(), config_candidate)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_candidate}")

    config = tu.load_config(config_path)
    device = torch.device(args.device) if args.device else default_device()

    enforced_horizons = parse_horizon_values(args.horizons) if args.horizons else None

    forecast_paths = _collect_paths(args.forecast_paths)
    chronos_paths = _collect_paths(args.chronos_paths)
    if not forecast_paths and not chronos_paths:
        raise ValueError("No checkpoints provided. Supply --forecast-paths or --chronos-paths")

    entries: List[CheckpointEntry] = []
    entries.extend(
        _prepare_checkpoint_entries(paths=forecast_paths, source="forecast", enforced_horizons=enforced_horizons, device=device)
    )
    entries.extend(
        _prepare_checkpoint_entries(paths=chronos_paths, source="chronos", enforced_horizons=enforced_horizons, device=device)
    )
    if not entries:
        raise ValueError("No valid checkpoints collected for evaluation")

    data_dir_default = (ROOT_DIR / "ICML_datasets").resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else data_dir_default

    module = TimeSeriesDataModule(
        dataset_name="",
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        normalize=True,
        filename=None,
        train=True,
        val=True,
        test=True,
    )
    dataset_groups = module.get_dataloaders()
    dataset_map: Dict[str, DatasetLoaders] = {group.name: group for group in dataset_groups}

    model_cfg = apply_model_overrides(
        config.model,
        token_size=args.token_size,
        model_dim=args.model_dim,
        embedding_dim=args.embedding_dim,
        depth=args.depth,
    )

    summary_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    for entry in entries:
        group = _find_dataset_group(dataset_map, entry)
        if group is None:
            print(f"Warning: dataset '{entry.dataset_name}' not available under {data_dir}; skipping {entry.path}")
            continue
        loader, split_name = _select_loader(group, args.split)
        if loader is None:
            print(f"Warning: dataset '{entry.dataset_name}' has no dataloader for requested split; skipping")
            continue
        try:
            model = _build_model(checkpoint=entry.checkpoint, horizons=entry.horizons, model_cfg=model_cfg, device=device)
        except Exception as exc:
            print(f"Warning: failed to build model for {entry.path}: {exc}")
            continue
        run_dir_path = entry.path.parent.parent
        run_dir_name = run_dir_path.name if run_dir_path != entry.path.parent else ""
        checkpoint_id = f"{run_dir_name}/{entry.path.parent.name}" if run_dir_name else entry.path.parent.name
        aggregates, samples = _evaluate_loader(
            model=model,
            loader=loader,
            horizons=entry.horizons,
            device=device,
            dataset_name=entry.dataset_name,
            split_name=split_name,
            checkpoint_id=checkpoint_id,
            source=entry.source,
        )
        summary_rows.extend(aggregates)
        sample_rows.extend(samples)

    if not sample_rows:
        print("No evaluation results were produced.")
        return

    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else (ROOT_DIR / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_name or "forecast_evaluation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_path = results_dir / f"{prefix}_samples_{timestamp}.csv"
    summary_path = results_dir / f"{prefix}_summary_{timestamp}.csv"

    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("Evaluation complete.")
    print(f"Samples CSV: {sample_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
