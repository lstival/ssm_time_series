"""Zero-shot evaluation of Chronos-trained forecasting heads on ICML datasets."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from util import default_device
from down_tasks.forecast_shared import apply_model_overrides
from down_tasks.forecast_utils import (
    ensure_dataloader_pred_len,
    print_evaluation_summary,
    save_evaluation_results,
)
from evaluation_down_tasks.zeroshot_utils import (
    build_model_from_checkpoint,
    dataset_slug,
    determine_config_path,
    evaluate_and_collect,
    extract_checkpoint_timestamp,
    load_zeroshot_config,
)
from evaluation_down_tasks.zeroshot_reporting import (
    aggregate_results_by_horizon,
    save_horizon_summary,
)
from data_provider.data_factory import data_provider as build_sd_data_provider
from models.utils import SDMambaDatasetConfig

CONFIG_ENV_VAR = "ICML_ZEROSHOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "icml_zeroshot.yaml"


def _context_length_from_sample_size(
    sample_size: Optional[Union[int, Sequence[int]]]
) -> Optional[int]:
    if sample_size is None:
        return None
    if isinstance(sample_size, int):
        return int(sample_size)
    values = list(sample_size)
    if not values:
        return None
    return int(values[0])


def _infer_target_column(file_path: Path, fallback: str = "OT") -> str:
    if file_path.suffix.lower() != ".csv":
        return fallback
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            header_line = handle.readline()
    except (OSError, UnicodeDecodeError):  # pragma: no cover - defensive
        return fallback
    header_line = header_line.lstrip("\ufeff").strip()
    if not header_line:
        return fallback
    columns = [col.strip() for col in header_line.split(",") if col.strip()]
    for column in columns:
        lower = column.lower()
        if lower not in {"date", "datetime"}:
            return column
    return fallback


def _discover_sd_mamba_datasets(
    data_dir: Path,
    dataset_names: Optional[Sequence[str]],
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
) -> List[Tuple[str, SDMambaDatasetConfig]]:
    entries: List[Tuple[str, SDMambaDatasetConfig]] = []
    filter_set = {name for name in dataset_names} if dataset_names else None
    processed_names: set[str] = set()

    def _add_entry(name: str, config: SDMambaDatasetConfig) -> None:
        entries.append((name, config))
        processed_names.add(config.data_path)

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return entries

    for parent in sorted(data_dir.iterdir()):
        if not parent.is_dir():
            continue
        parent_key = parent.name.lower()
        if parent_key == "ett-small":
            for file_path in sorted(parent.glob("*.csv")):
                if filter_set and file_path.name not in filter_set:
                    continue
                stem = file_path.stem
                freq = "h" if "h" in stem.lower() else "t"
                config = SDMambaDatasetConfig(
                    name=stem,
                    data_key=stem,
                    root_path=parent,
                    data_path=file_path.name,
                    features="S",
                    target="OT",
                    freq=freq,
                    seq_len=seq_len,
                    label_len=0,
                    pred_len=pred_len,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    embed="timeF",
                    scale=True,
                    scaler_type="standard",
                    train_shuffle=True,
                    train_drop_last=True,
                    val_shuffle=False,
                    val_drop_last=False,
                )
                _add_entry(stem, config)
        elif parent_key == "pems":
            for file_path in sorted(parent.glob("*.npz")):
                if filter_set and file_path.name not in filter_set:
                    continue
                stem = file_path.stem
                config = SDMambaDatasetConfig(
                    name=stem,
                    data_key="PEMS",
                    root_path=parent,
                    data_path=file_path.name,
                    features="S",
                    target="OT",
                    freq="h",
                    seq_len=seq_len,
                    label_len=0,
                    pred_len=pred_len,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    embed="timeF",
                    scale=True,
                    scaler_type="standard",
                    train_shuffle=True,
                    train_drop_last=True,
                    val_shuffle=False,
                    val_drop_last=False,
                )
                _add_entry(stem, config)
        elif parent_key == "solar":
            for file_path in sorted(parent.glob("*.txt")):
                if filter_set and file_path.name not in filter_set:
                    continue
                stem = file_path.stem
                config = SDMambaDatasetConfig(
                    name=stem,
                    data_key="Solar",
                    root_path=parent,
                    data_path=file_path.name,
                    features="S",
                    target="OT",
                    freq="h",
                    seq_len=seq_len,
                    label_len=0,
                    pred_len=pred_len,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    embed="timeF",
                    scale=True,
                    scaler_type="standard",
                    train_shuffle=True,
                    train_drop_last=True,
                    val_shuffle=False,
                    val_drop_last=False,
                )
                _add_entry(stem, config)
        elif parent_key in {"electricity", "exchange_rate", "traffic", "weather"}:
            for file_path in sorted(parent.glob("*.csv")):
                if filter_set and file_path.name not in filter_set:
                    continue
                stem = file_path.stem
                target = _infer_target_column(file_path)
                freq_map = {
                    "electricity": "h",
                    "exchange_rate": "d",
                    "traffic": "h",
                    "weather": "h",
                }
                config = SDMambaDatasetConfig(
                    name=stem,
                    data_key="custom",
                    root_path=parent,
                    data_path=file_path.name,
                    features="S",
                    target=target,
                    freq=freq_map.get(parent_key, "h"),
                    seq_len=seq_len,
                    label_len=0,
                    pred_len=pred_len,
                    batch_size=batch_size,
                    val_batch_size=val_batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    embed="timeF",
                    scale=True,
                    scaler_type="standard",
                    train_shuffle=True,
                    train_drop_last=True,
                    val_shuffle=False,
                    val_drop_last=False,
                )
                _add_entry(stem, config)

    if filter_set:
        missing = sorted(filter_set - processed_names)
        if missing:
            print(f"Warning: requested datasets not found in {data_dir}: {missing}")

    entries.sort(key=lambda item: item[0].lower())
    return entries


def _build_provider_args(config: SDMambaDatasetConfig, *, batch_size: int) -> SimpleNamespace:
    payload = {
        "root_path": str(config.root_path),
        "data_path": config.data_path,
        "data": config.data_key,
        "features": config.features,
        "target": config.target,
        "freq": config.freq,
        "seq_len": config.seq_len,
        "label_len": config.label_len,
        "pred_len": config.pred_len,
        "batch_size": batch_size,
        "val_batch_size": config.val_batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": False,
        "embed": config.embed,
        "scale": config.scale,
        "scaler_type": config.scaler_type,
        "train_shuffle": config.train_shuffle,
        "train_drop_last": config.train_drop_last,
        "val_shuffle": config.val_shuffle,
        "val_drop_last": config.val_drop_last,
        "test_batch_size": config.val_batch_size,
        "test_shuffle": False,
        "test_drop_last": False,
        "pred_batch_size": config.val_batch_size,
        "pred_shuffle": False,
        "pred_drop_last": False,
    }
    return SimpleNamespace(**payload)



if __name__ == "__main__":
    config_path = determine_config_path(DEFAULT_CONFIG_PATH)
    zeroshot_cfg = load_zeroshot_config(config_path)

    base_config = tu.load_config(zeroshot_cfg.model_config_path)

    seed = zeroshot_cfg.seed if zeroshot_cfg.seed is not None else base_config.seed
    tu.set_seed(seed)
    torch.manual_seed(seed)

    horizons = zeroshot_cfg.horizons
    context_length = _context_length_from_sample_size(zeroshot_cfg.sample_size)
    effective_prefix = zeroshot_cfg.output_prefix
    if context_length is not None:
        effective_prefix = f"{effective_prefix}_input{context_length}"
    device = default_device()
    print(f"Using device: {device}")
    print(f"Requested horizons: {horizons}")
    print(f"Using zero-shot configuration: {zeroshot_cfg.config_path}")
    print(f"Using encoder configuration: {zeroshot_cfg.model_config_path}")
    if context_length is not None:
        print(f"Using context length override: {context_length}")

    results_dir = zeroshot_cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = apply_model_overrides(
        base_config.model,
        token_size=zeroshot_cfg.overrides.get("token_size"),
        model_dim=zeroshot_cfg.overrides.get("model_dim"),
        embedding_dim=zeroshot_cfg.overrides.get("embedding_dim"),
        depth=zeroshot_cfg.overrides.get("depth"),
    )

    model, checkpoint_info, eval_horizons, max_horizon, sequence_first_input = build_model_from_checkpoint(
        model_cfg=model_cfg,
        checkpoint_path=zeroshot_cfg.forecast_checkpoint_path,
        requested_horizons=horizons,
        device=device,
        encoder_checkpoint_path=zeroshot_cfg.encoder_checkpoint_path,
        visual_encoder_checkpoint_path=None,
        force_dual=False,
    )

    checkpoint_info.update(
        {
            "encoder_config_path": str(zeroshot_cfg.model_config_path),
            "encoder_checkpoint_path": str(
                zeroshot_cfg.encoder_checkpoint_path or zeroshot_cfg.forecast_checkpoint_path
            ),
            "forecast_head_checkpoint_path": str(zeroshot_cfg.forecast_checkpoint_path),
            "zero_shot_config_path": str(zeroshot_cfg.config_path),
        }
    )

    print(
        "Loaded forecasting model with horizons "
        f"{checkpoint_info['horizons']} (evaluating {eval_horizons}), max horizon {max_horizon}"
    )

    # Generate timestamp from checkpoint name or current time
    checkpoint_timestamp = extract_checkpoint_timestamp(zeroshot_cfg.forecast_checkpoint_path)
    if checkpoint_timestamp is None:
        checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create predictions directory early
    predictions_dir = results_dir / f"{effective_prefix}_{checkpoint_timestamp}_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    seq_len = context_length if context_length is not None else 96
    pred_len_required = max(max_horizon, max(eval_horizons)) if eval_horizons else max_horizon

    dataset_entries = _discover_sd_mamba_datasets(
        Path(zeroshot_cfg.data_dir),
        zeroshot_cfg.dataset_names,
        seq_len=seq_len,
        pred_len=pred_len_required,
        batch_size=zeroshot_cfg.batch_size,
        val_batch_size=zeroshot_cfg.val_batch_size,
        num_workers=zeroshot_cfg.num_workers,
    )
    if not dataset_entries:
        raise RuntimeError("No datasets available for evaluation. Check filters or data directory.")

    results_by_dataset: Dict[str, Dict[int, Dict[str, float]]] = {}
    prediction_payloads: Dict[str, Dict[str, object]] = {}

    for display_name, sd_config in dataset_entries:
        data_source = sd_config.root_path / sd_config.data_path
        print(f"\nPreparing dataset '{display_name}' from {data_source}...")

        if zeroshot_cfg.split == "all":
            flags_to_fetch = ["train", "val", "test"]
        elif zeroshot_cfg.split == "test":
            flags_to_fetch = ["test", "val"]
        else:
            flags_to_fetch = ["val", "test"]

        loaded: Dict[str, Tuple[object, DataLoader]] = {}
        for flag in flags_to_fetch:
            batch_for_flag = sd_config.batch_size if flag == "train" else sd_config.val_batch_size
            args = _build_provider_args(sd_config, batch_size=batch_for_flag)
            try:
                dataset_obj, loader = build_sd_data_provider(args, flag)
            except FileNotFoundError as exc:
                print(f"  {flag.title()} split unavailable for '{display_name}': {exc}")
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"  Failed to prepare {flag} split for '{display_name}': {exc}")
                continue

            if loader is not None:
                loaded[flag] = (dataset_obj, loader)

        eval_loader: Optional[DataLoader] = None
        eval_dataset = None
        split_used = zeroshot_cfg.split

        if zeroshot_cfg.split == "all":
            component_datasets = [entry[0] for entry in loaded.values()]
            if component_datasets:
                concat_dataset = ConcatDataset(component_datasets)
                eval_loader = DataLoader(
                    concat_dataset,
                    batch_size=sd_config.val_batch_size,
                    shuffle=False,
                    num_workers=sd_config.num_workers,
                    pin_memory=sd_config.pin_memory,
                    persistent_workers=False,
                )
                eval_dataset = component_datasets[0]
                used_splits = "+".join(flag for flag in ["train", "val", "test"] if flag in loaded)
                print(f"  Using splits: {used_splits or 'none'}")
                split_used = "all"
            else:
                print(f"  No splits available for '{display_name}'. Skipping.")
                continue
        else:
            preferred = "test" if zeroshot_cfg.split == "test" else "val"
            fallback = "val" if preferred == "test" else "test"

            if preferred in loaded:
                eval_dataset, eval_loader = loaded[preferred]
                split_used = preferred
            elif fallback in loaded:
                eval_dataset, eval_loader = loaded[fallback]
                print(
                    f"  Preferred split '{preferred}' unavailable for '{display_name}'; using '{fallback}' instead."
                )
                split_used = fallback
            else:
                print(
                    f"  Skipping dataset '{display_name}' because neither preferred nor fallback splits are available."
                )
                continue

        if eval_loader is None:
            print(f"  No evaluation loader constructed for '{display_name}'.")
            continue

        try:
            ensure_dataloader_pred_len(eval_loader, max_horizon)
            print(f"Evaluating dataset '{display_name}' on '{split_used}' split...")
            metrics, payload = evaluate_and_collect(
                model,
                eval_loader,
                device,
                list(eval_horizons),
                max_horizon,
                sequence_first_input=sequence_first_input,
                dataset=eval_dataset,
                dataset_name=display_name,
            )
            if metrics is None or payload is None:
                print(f"  No metrics computed for dataset '{display_name}'.")
                continue

            results_by_dataset[display_name] = metrics
            payload["dataset"] = display_name
            payload["dataset_slug"] = dataset_slug(display_name)
            payload["split"] = split_used
            prediction_payloads[display_name] = payload

            dataset_dir = predictions_dir / payload["dataset_slug"]
            dataset_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, dataset_dir / "data.pt")
            print(f"  Saved predictions to: {dataset_dir / 'data.pt'}")

            for horizon in eval_horizons:
                if horizon in metrics:
                    mse = metrics[horizon]["mse"]
                    mae = metrics[horizon]["mae"]
                    mse_str = f"{mse:.6f}" if not math.isnan(mse) else "nan"
                    mae_str = f"{mae:.6f}" if not math.isnan(mae) else "nan"
                    print(f"  H{horizon}: MSE={mse_str}, MAE={mae_str}")

        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                print(f"  Skipping dataset '{display_name}' due to out of memory error: {exc}")
                torch.cuda.empty_cache()
                continue
            raise

    if not results_by_dataset:
        print("No datasets produced evaluation metrics. Nothing to report.")
        sys.exit(0)

    dataset_json_path, dataset_csv_path = save_evaluation_results(
        results_by_dataset,
        checkpoint_info,
        results_dir,
        prefix=effective_prefix,
        timestamp=checkpoint_timestamp,
    )

    print_evaluation_summary(results_by_dataset, checkpoint_info)

    horizon_summary = aggregate_results_by_horizon(results_by_dataset)
    horizon_json_path, horizon_csv_path = save_horizon_summary(
        horizon_summary,
        results_dir=results_dir,
        prefix=effective_prefix,
        timestamp=checkpoint_timestamp,
    )

    print("\nSaved evaluation artifacts:")
    print(f"  Per-dataset JSON: {dataset_json_path}")
    print(f"  Per-dataset CSV:  {dataset_csv_path}")
    print(f"  Per-horizon JSON: {horizon_json_path}")
    print(f"  Per-horizon CSV:  {horizon_csv_path}")
    print(f"  Predictions dir:  {predictions_dir}")

    print("\nModel artifact references:")
    print(f"  Encoder config: {zeroshot_cfg.model_config_path}")
    if zeroshot_cfg.encoder_checkpoint_path is not None:
        print(f"  Encoder checkpoint: {zeroshot_cfg.encoder_checkpoint_path}")
    if zeroshot_cfg.visual_encoder_checkpoint_path is not None:
        print(f"  Visual encoder checkpoint: {zeroshot_cfg.visual_encoder_checkpoint_path}")
    print(f"  Forecast head checkpoint: {zeroshot_cfg.forecast_checkpoint_path}")

    print("\nPer-horizon summary (mean metrics across datasets):")
    for horizon in sorted(horizon_summary.keys()):
        payload = horizon_summary[horizon]
        mse = payload["mean_mse"]
        mae = payload["mean_mae"]
        mse_str = f"{mse:.6f}" if not math.isnan(mse) else "nan"
        mae_str = f"{mae:.6f}" if not math.isnan(mae) else "nan"
        print(
            f"  H{horizon}: datasets={payload['dataset_count']}, "
            f"mean MSE={mse_str}, mean MAE={mae_str}"
        )
