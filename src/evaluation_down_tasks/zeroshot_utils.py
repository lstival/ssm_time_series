"""Shared utilities for zero-shot ICML forecasting evaluation and plotting."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from moco_training import resolve_path
from down_tasks.forecast_shared import parse_horizon_values
from models.classifier import ForecastRegressor, MultiHorizonForecastMLP


@dataclass
class ZeroShotConfig:
    config_path: Path
    model_config_path: Path
    overrides: Dict[str, object]
    forecast_checkpoint_path: Path
    encoder_checkpoint_path: Optional[Path]
    results_dir: Path
    data_dir: str
    dataset_name: str
    filename: Optional[str]
    batch_size: int
    val_batch_size: int
    num_workers: int
    horizons: List[int]
    split: str
    output_prefix: str
    seed: Optional[int]


@dataclass
class PlotConfig:
    config_path: Path
    predictions_root: Optional[Path]
    dataset_names: List[str]
    dataset_files: Dict[str, Path]
    horizon_list: List[int]
    sample_indices: List[int]
    feature_indices: List[int]
    output_dir: Path
    show: bool
    dpi: int
    figsize: Sequence[float]


def dataset_slug(name: str) -> str:
    slug = name.replace("\\", "__").replace("/", "__").strip()
    return slug or "dataset"


def extract_checkpoint_timestamp(checkpoint_path: Path) -> Optional[str]:
    matches = re.findall(r"\d{8}_\d{4}", str(checkpoint_path))
    if matches:
        return matches[-1]
    return None


def normalize_horizons(raw: object) -> List[int]:
    if isinstance(raw, str):
        return parse_horizon_values(raw)
    try:
        values = [int(item) for item in raw]  # type: ignore[iterable]
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValueError("'horizons' must be a list of integers or comma-separated string") from exc
    if not values:
        raise ValueError("At least one horizon must be provided in configuration")
    return sorted(set(values))


def resolve_required_path(base: Path, candidate: object, *, description: str) -> Path:
    if candidate is None:
        raise ValueError(f"Configuration missing required path for {description}")
    resolved = resolve_path(base, Path(candidate))
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {candidate}")
    return resolved


def resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    if candidate is None:
        return None
    resolved = resolve_path(base, Path(candidate))
    return resolved


def determine_config_path(default_path: Path) -> Path:
    resolved = resolve_path(Path.cwd(), default_path)
    if resolved is None or not resolved.exists():
        resolved = default_path.expanduser().resolve()
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {default_path}")
    return resolved


def load_zeroshot_config(config_path: Path) -> ZeroShotConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    config_dir = config_path.parent

    model_section = dict(payload.get("model") or {})
    model_config_path = resolve_required_path(
        config_dir,
        model_section.get("config"),
        description="model configuration",
    )
    overrides = dict(model_section.get("overrides") or {})

    paths_section = dict(payload.get("paths") or {})
    forecast_checkpoint_candidate = paths_section.get("forecast_checkpoint")
    forecast_checkpoint_path = resolve_required_path(
        config_dir,
        forecast_checkpoint_candidate,
        description="forecast head checkpoint",
    )
    encoder_checkpoint_path = resolve_optional_path(config_dir, paths_section.get("encoder_checkpoint"))

    results_candidate = paths_section.get("results_dir")
    if results_candidate is None:
        results_dir = (ROOT_DIR / "results").resolve()
    else:
        results_dir = resolve_path(config_dir, Path(results_candidate))
        if results_dir is None:
            results_dir = Path(results_candidate).expanduser().resolve()

    data_section = dict(payload.get("data") or {})
    data_dir_value = data_section.get("data_dir")
    if data_dir_value is None:
        data_dir = str((ROOT_DIR / "ICML_datasets").resolve())
    else:
        resolved_data_dir = resolve_path(config_dir, Path(data_dir_value))
        if resolved_data_dir is None:
            resolved_data_dir = Path(data_dir_value).expanduser().resolve()
        data_dir = str(resolved_data_dir)

    dataset_name = str(data_section.get("dataset_name", "") or "")
    filename_value = data_section.get("filename")
    if filename_value:
        resolved_filename = resolve_path(config_dir, Path(filename_value))
        if resolved_filename is not None:
            filename = str(resolved_filename)
        else:
            filename = str(Path(filename_value).expanduser().resolve())
    else:
        filename = None

    batch_size = int(data_section.get("batch_size", 32))
    val_batch_size = int(data_section.get("val_batch_size", batch_size))
    num_workers = int(data_section.get("num_workers", 4))

    evaluation_section = dict(payload.get("evaluation") or {})
    horizons = normalize_horizons(evaluation_section.get("horizons", [96, 192, 336, 720]))
    split = str(evaluation_section.get("split", "test")).lower()
    if split not in {"test", "val", "all"}:
        raise ValueError("'evaluation.split' must be either 'test', 'all', or 'val'")
    output_prefix = str(evaluation_section.get("output_prefix", "icml_zeroshot_forecast"))

    seed_value = payload.get("seed")
    if seed_value is None:
        seed_value = evaluation_section.get("seed")
    seed = int(seed_value) if seed_value is not None else None

    results_dir = results_dir.expanduser().resolve()

    return ZeroShotConfig(
        config_path=config_path,
        model_config_path=model_config_path,
        overrides=overrides,
        forecast_checkpoint_path=forecast_checkpoint_path,
        encoder_checkpoint_path=encoder_checkpoint_path,
        results_dir=results_dir,
        data_dir=data_dir,
        dataset_name=dataset_name,
        filename=filename,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        horizons=horizons,
        split=split,
        output_prefix=output_prefix,
        seed=seed,
    )


def evaluate_and_collect(
    model: ForecastRegressor,
    loader,
    device: torch.device,
    horizons: Sequence[int],
    max_horizon: int,
) -> Tuple[Optional[Dict[int, Dict[str, float]]], Optional[Dict[str, object]]]:
    if loader is None:
        return None, None

    model.eval()
    running_sse = {int(h): 0.0 for h in horizons}
    running_sae = {int(h): 0.0 for h in horizons}
    running_ape = {int(h): 0.0 for h in horizons}
    counts = {int(h): 0 for h in horizons}
    total_samples = 0

    contexts: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    predictions_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            seq_x, seq_y, _, _ = batch
            seq_x_cpu = seq_x.float()
            seq_x_enc = seq_x_cpu.to(device).transpose(1, 2)
            seq_y_cpu = seq_y.float()

            if seq_y_cpu.size(1) < max_horizon:
                raise ValueError(
                    f"Target sequence length {seq_y_cpu.size(1)} smaller than required max horizon {max_horizon}"
                )

            target_slice = seq_y_cpu[:, -max_horizon:, :].to(device)
            preds = model(seq_x_enc)
            batch_size = target_slice.size(0)
            total_samples += batch_size

            epsilon = 1e-8
            for horizon in horizons:
                horizon = int(horizon)
                pred_sub = preds[:, :horizon, :]
                target_sub = target_slice[:, :horizon, :]
                diff = pred_sub - target_sub

                sse = torch.sum(diff.pow(2)).item()
                sae = torch.sum(torch.abs(diff)).item()
                ape = torch.sum(torch.abs(diff) / (torch.abs(target_sub) + epsilon)).item()
                elements = diff.numel()

                running_sse[horizon] += sse
                running_sae[horizon] += sae
                running_ape[horizon] += ape
                counts[horizon] += elements

            contexts.append(seq_x_cpu)
            targets_list.append(target_slice.cpu())
            predictions_list.append(preds.cpu())

    if not contexts:
        return None, None

    results: Dict[int, Dict[str, float]] = {}
    for horizon in horizons:
        horizon = int(horizon)
        if counts[horizon] == 0:
            results[horizon] = {
                "mse": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "samples": total_samples,
            }
        else:
            mse = running_sse[horizon] / counts[horizon]
            mae = running_sae[horizon] / counts[horizon]
            rmse = math.sqrt(mse)
            mape = (running_ape[horizon] / counts[horizon]) * 100.0
            results[horizon] = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "samples": total_samples,
            }

    context_tensor = torch.cat(contexts, dim=0).contiguous()
    targets_tensor = torch.cat(targets_list, dim=0).contiguous()
    predictions_tensor = torch.cat(predictions_list, dim=0).contiguous()

    per_horizon: Dict[int, Dict[str, torch.Tensor]] = {}
    for horizon in horizons:
        horizon = int(horizon)
        per_horizon[horizon] = {
            "targets": targets_tensor[:, :horizon, :].clone(),
            "predictions": predictions_tensor[:, :horizon, :].clone(),
        }

    payload = {
        "context": context_tensor.cpu(),
        "targets": targets_tensor.cpu(),
        "predictions": predictions_tensor.cpu(),
        "eval_horizons": [int(h) for h in horizons],
        "max_horizon": int(max_horizon),
        "context_length": int(context_tensor.size(1)),
        "target_features": int(context_tensor.size(2)),
        "per_horizon": {h: {key: tensor.cpu() for key, tensor in values.items()} for h, values in per_horizon.items()},
    }

    return results, payload


def build_model_from_checkpoint(
    *,
    model_cfg: Dict[str, object],
    checkpoint_path: Path,
    requested_horizons: Sequence[int],
    device: torch.device,
) -> Tuple[ForecastRegressor, Dict[str, object], List[int], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        encoder = tu.build_encoder_from_config(model_cfg).to(device)
    except Exception:
        encoder = tu.build_visual_encoder_from_config(model_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    head_weight_key = "head.shared_layers.0.weight"
    if head_weight_key not in state_dict:
        raise KeyError(
            "Checkpoint missing head weights; expected unified multi-horizon head entries."
        )

    hidden_dim = state_dict[head_weight_key].shape[0]
    input_dim = state_dict[head_weight_key].shape[1]
    if hasattr(encoder, "embedding_dim") and int(encoder.embedding_dim) != int(input_dim):
        raise ValueError(
            "Encoder embedding dimension does not match head input dimension: "
            f"{encoder.embedding_dim} != {input_dim}"
        )

    ckpt_horizons = checkpoint.get("horizons")
    if ckpt_horizons is None:
        ckpt_horizons = list(requested_horizons)
    ckpt_horizons = sorted({int(h) for h in ckpt_horizons})

    requested_set = [int(h) for h in requested_horizons]
    evaluation_horizons = [h for h in requested_set if h in ckpt_horizons]
    if not evaluation_horizons:
        evaluation_horizons = list(ckpt_horizons)
        print(
            "Warning: requested horizons not found in checkpoint; falling back to checkpoint horizons: "
            f"{evaluation_horizons}"
        )

    target_features = int(checkpoint.get("target_features", 1))
    max_horizon = int(checkpoint.get("max_horizon", max(ckpt_horizons)))

    head = MultiHorizonForecastMLP(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        horizons=list(ckpt_horizons),
        target_features=target_features,
    ).to(device)

    model = ForecastRegressor(encoder=encoder, head=head, freeze_encoder=True).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing weights when loading checkpoint: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected weights when loading checkpoint: {sorted(unexpected)}")

    checkpoint_info = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "avg_val_mse": checkpoint.get("avg_val_mse"),
        "train_losses": checkpoint.get("train_losses"),
        "val_results": checkpoint.get("val_results"),
        "horizons": list(ckpt_horizons),
        "target_features": target_features,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "max_horizon": max_horizon,
        "encoder_embedding_dim": getattr(encoder, "embedding_dim", None),
    }
    return model, checkpoint_info, list(evaluation_horizons), max_horizon


def select_loader(group, preferred_split: str):
    def _resolve(split_name: str, loader):
        dataset_attr = f"{split_name}_dataset"
        dataset_obj = getattr(group, dataset_attr, None)
        if dataset_obj is None and loader is not None:
            dataset_obj = getattr(loader, "dataset", None)
        return loader, dataset_obj

    if preferred_split == "all":
        # Concatenate datasets from train, val, and test splits
        from torch.utils.data import ConcatDataset
        
        datasets = []
        available_splits = []
        
        for split_name, loader in [("train", group.train), ("val", group.val), ("test", group.test)]:
            if loader is not None:
                _, dataset = _resolve(split_name, loader)
                if dataset is not None:
                    datasets.append(dataset)
                    available_splits.append(split_name)
        
        if not datasets:
            return None, None, "all"
        
        concatenated_dataset = ConcatDataset(datasets)
        split_names = "+".join(available_splits)
        print(f"  Using concatenated dataset from splits: {split_names}")
        
        # Return the first available loader (structure), concatenated dataset, and "all" indicator
        first_loader = next((loader for loader in [group.train, group.val, group.test] if loader is not None), None)
        return first_loader, concatenated_dataset, "all"
    
    if preferred_split == "test":
        primary_loader, fallback_loader = group.test, group.val
        primary_name, fallback_name = "test", "val"
    else:
        primary_loader, fallback_loader = group.val, group.test
        primary_name, fallback_name = "val", "test"

    if primary_loader is not None:
        loader, dataset = _resolve(primary_name, primary_loader)
        return loader, dataset, primary_name
    if fallback_loader is not None:
        loader, dataset = _resolve(fallback_name, fallback_loader)
        print(
            f"  Preferred split '{primary_name}' unavailable for {group.name}; using '{fallback_name}' split instead."
        )
        return loader, dataset, fallback_name
    return None, None, primary_name


def list_dataset_directories(root: Path) -> List[str]:
    return sorted([entry.name for entry in root.iterdir() if entry.is_dir()])


def ensure_sequence(value: Optional[Iterable[int]], *, default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, (int, float)):
        return [int(value)]
    return [int(item) for item in value]


def load_plot_config(config_path: Path) -> PlotConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    predictions_section = dict(payload.get("predictions") or {})
    root_dir_value = predictions_section.get("root_dir")
    file_value = predictions_section.get("file")

    dataset_files: Dict[str, Path] = {}
    dataset_names: List[str] = []
    predictions_root: Optional[Path] = None

    if file_value is not None:
        predictions_file = Path(file_value).expanduser().resolve()
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        predictions_root = predictions_file.parent.parent
        dataset_label = str(predictions_section.get("dataset") or predictions_file.parent.name)
        dataset_names = [dataset_label]
        dataset_files[dataset_label] = predictions_file
    else:
        if root_dir_value is None:
            raise ValueError("Configuration must provide either 'predictions.root_dir' or 'predictions.file'.")
        predictions_root = Path(root_dir_value).expanduser().resolve()
        if not predictions_root.exists():
            raise FileNotFoundError(f"Predictions root directory not found: {predictions_root}")

        dataset_value = predictions_section.get("dataset")
        if dataset_value is None:
            configured_names: List[str] = []
        elif isinstance(dataset_value, (list, tuple)):
            configured_names = [str(item) for item in dataset_value if str(item).strip()]
        else:
            configured_names = [str(dataset_value).strip()] if str(dataset_value).strip() else []

        if not configured_names or configured_names == ["dataset"]:
            discovered = list_dataset_directories(predictions_root)
            if not discovered:
                raise ValueError(f"No dataset directories found under {predictions_root}")
            candidate_names = discovered
        else:
            candidate_names = configured_names

        resolved_names: List[str] = []
        for dataset_name in candidate_names:
            candidates = [predictions_root / dataset_name]
            slug = dataset_slug(dataset_name)
            if slug != dataset_name:
                candidates.append(predictions_root / slug)
            dataset_dir = next((path for path in candidates if path.exists()), None)
            if dataset_dir is None:
                raise FileNotFoundError(
                    f"Dataset directory '{dataset_name}' not found under {predictions_root}"
                )
            predictions_file = dataset_dir / "data.pt"
            if not predictions_file.exists():
                raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
            label = dataset_dir.name
            resolved_names.append(label)
            dataset_files[label] = predictions_file

        dataset_names = resolved_names

    plot_section = dict(payload.get("plot") or {})
    horizons = ensure_sequence(plot_section.get("horizons"), default=[96])
    sample_indices = ensure_sequence(plot_section.get("sample_indices"), default=[0])
    feature_indices = ensure_sequence(plot_section.get("feature_indices"), default=[0])

    output_dir_value = plot_section.get("output_dir")
    output_dir = (
        Path(output_dir_value).expanduser().resolve()
        if output_dir_value is not None
        else ROOT_DIR / "results" / "plots"
    )

    show = bool(plot_section.get("show", False))
    dpi = int(plot_section.get("dpi", 150))
    figsize = plot_section.get("figsize", [12, 6])
    if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
        raise ValueError("'plot.figsize' must be a list or tuple of length 2")

    return PlotConfig(
        config_path=config_path,
        predictions_root=predictions_root,
        dataset_names=dataset_names,
        dataset_files=dataset_files,
        horizon_list=horizons,
        sample_indices=sample_indices,
        feature_indices=feature_indices,
        output_dir=output_dir,
        show=show,
        dpi=dpi,
        figsize=figsize,
    )
