"""Zero-shot evaluation for Chronos-supervised encoders on ICML datasets."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from chronos_supervised_training import ChronosForecastModel
from time_series_loader import TimeSeriesDataModule
from util import default_device
from down_tasks.forecast_utils import (
    ensure_dataloader_pred_len,
    print_evaluation_summary,
    save_evaluation_results,
)
from evaluation_down_tasks.zeroshot_reporting import (
    aggregate_results_by_horizon,
    save_horizon_summary,
)
from evaluation_down_tasks.zeroshot_utils import (
    dataset_slug,
    determine_config_path,
    evaluate_and_collect,
    extract_checkpoint_timestamp,
    normalize_horizons,
    resolve_optional_path,
    resolve_required_path,
    select_loader,
)

CONFIG_ENV_VAR = "CHRONOS_SUPERVISED_ZEROSHOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "chronos_supervised_zeroshot.yaml"


@dataclass
class ChronosSupervisedZeroShotConfig:
    config_path: Path
    training_config_path: Path
    checkpoint_path: Path
    encoder_type: str
    results_dir: Path
    data_dir: str
    dataset_name: str
    dataset_names: Optional[List[str]]
    filename: Optional[str]
    batch_size: int
    val_batch_size: int
    num_workers: int
    horizons: List[int]
    split: str
    output_prefix: str
    seed: Optional[int]


def load_chronos_supervised_config(config_path: Path) -> ChronosSupervisedZeroShotConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    config_dir = config_path.parent

    training_config_candidate = payload.get("training_config")
    training_config_path = resolve_required_path(
        config_dir,
        training_config_candidate,
        description="Chronos supervised training config",
    )

    encoder_type_candidate = payload.get("encoder_type")
    if encoder_type_candidate is None:
        encoder_type_candidate = (payload.get("model") or {}).get("encoder_type")
    encoder_type = str(encoder_type_candidate or "temporal").strip().lower()
    if encoder_type not in {"temporal", "visual"}:
        raise ValueError("'encoder_type' must be either 'temporal' or 'visual'")

    paths_section = dict(payload.get("paths") or {})
    checkpoint_path = resolve_required_path(
        config_dir,
        paths_section.get("checkpoint"),
        description="Chronos supervised checkpoint",
    )
    results_candidate = paths_section.get("results_dir")
    if results_candidate is None:
        results_dir = (ROOT_DIR / "results").resolve()
    else:
        resolved = resolve_optional_path(config_dir, results_candidate)
        results_dir = resolved if resolved is not None else Path(results_candidate).expanduser().resolve()

    data_section = dict(payload.get("data") or {})
    data_dir_value = data_section.get("data_dir")
    if data_dir_value is None:
        data_dir = str((ROOT_DIR / "ICML_datasets").resolve())
    else:
        resolved_data_dir = resolve_optional_path(config_dir, data_dir_value)
        if resolved_data_dir is None:
            resolved_data_dir = Path(data_dir_value).expanduser().resolve()
        data_dir = str(resolved_data_dir)

    dataset_name = str(data_section.get("dataset_name", "") or "")
    dataset_names_value = data_section.get("dataset_names")
    dataset_names = None
    if dataset_names_value and isinstance(dataset_names_value, list):
        dataset_names = [str(name) for name in dataset_names_value]

    filename_value = data_section.get("filename")
    if filename_value:
        resolved_filename = resolve_optional_path(config_dir, filename_value)
        filename = str(resolved_filename) if resolved_filename is not None else str(Path(filename_value).expanduser().resolve())
    else:
        filename = None

    batch_size = int(data_section.get("batch_size", 512))
    val_batch_size = int(data_section.get("val_batch_size", batch_size))
    num_workers = int(data_section.get("num_workers", 4))

    evaluation_section = dict(payload.get("evaluation") or {})
    horizons = normalize_horizons(evaluation_section.get("horizons", [96, 192, 336, 720]))
    split = str(evaluation_section.get("split", "test")).lower()
    if split not in {"test", "val", "all"}:
        raise ValueError("'evaluation.split' must be either 'test', 'val', or 'all'")
    output_prefix = str(evaluation_section.get("output_prefix", "chronos_supervised_zeroshot"))

    seed_value = payload.get("seed")
    if seed_value is None:
        seed_value = evaluation_section.get("seed")
    seed = int(seed_value) if seed_value is not None else None

    results_dir = results_dir.expanduser().resolve()

    return ChronosSupervisedZeroShotConfig(
        config_path=config_path,
        training_config_path=training_config_path,
        checkpoint_path=checkpoint_path,
        encoder_type=encoder_type,
        results_dir=results_dir,
        data_dir=data_dir,
        dataset_name=dataset_name,
        dataset_names=dataset_names,
        filename=filename,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        horizons=horizons,
        split=split,
        output_prefix=output_prefix,
        seed=seed,
    )


def _resolve_state_dict(payload: Dict[str, object]) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate  # type: ignore[return-value]
    if all(isinstance(k, str) for k in payload.keys()) and all(
        isinstance(v, torch.Tensor) for v in payload.values()
    ):
        return payload  # type: ignore[return-value]
    raise KeyError("Checkpoint does not contain model weights under 'model_state'.")


def build_chronos_model(
    *,
    checkpoint_path: Path,
    training_config: tu.ExperimentConfig,
    requested_horizons: Sequence[int],
    encoder_type: str,
    device: torch.device,
) -> Tuple[ChronosForecastModel, Dict[str, object], List[int], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _resolve_state_dict(checkpoint)

    config_encoder = str(training_config.training.get("encoder_type", encoder_type)).lower()
    encoder_choice = encoder_type or config_encoder
    if encoder_choice not in {"temporal", "visual"}:
        raise ValueError(f"Unsupported encoder_type '{encoder_choice}'. Use 'temporal' or 'visual'.")
    if config_encoder in {"temporal", "visual"} and config_encoder != encoder_choice:
        print(
            f"Warning: zero-shot encoder_type '{encoder_choice}' differs from training config '{config_encoder}'."
        )
    pred_len = int(training_config.training.get("pred_len", 1))
    if pred_len <= 0:
        raise ValueError("training.pred_len must be positive in the Chronos supervised config")

    channel_weights = state_dict.get("channel_adapter.weight")
    if channel_weights is None:
        raise KeyError("Checkpoint missing 'channel_adapter.weight'; incompatible Chronos checkpoint")
    input_features = int(channel_weights.shape[1])

    head_weights = state_dict.get("head.weight")
    if head_weights is None:
        raise KeyError("Checkpoint missing 'head.weight'; incompatible Chronos checkpoint")
    target_dim = head_weights.shape[0] // pred_len
    if target_dim * pred_len != head_weights.shape[0]:
        raise ValueError("Head weight rows must be divisible by pred_len to infer target dimension")

    if encoder_choice == "visual":
        encoder = tu.build_visual_encoder_from_config(training_config.model)
    else:
        encoder = tu.build_encoder_from_config(training_config.model)

    model = ChronosForecastModel(
        encoder=encoder,
        input_features=input_features,
        target_dim=target_dim,
        pred_len=pred_len,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing weights when loading Chronos checkpoint: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected weights when loading Chronos checkpoint: {sorted(unexpected)}")

    max_horizon = pred_len
    eval_horizons = [int(h) for h in requested_horizons if int(h) <= max_horizon]
    if not eval_horizons:
        raise ValueError(
            f"Requested horizons {list(requested_horizons)} exceed model prediction length {max_horizon}"
        )

    checkpoint_info = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "train_loss": checkpoint.get("train_loss"),
        "val_loss": checkpoint.get("val_loss"),
        "pred_len": pred_len,
        "horizons": eval_horizons,
        "input_features": input_features,
        "target_features": target_dim,
        "encoder_type": encoder_choice,
    }

    return model, checkpoint_info, eval_horizons, max_horizon


if __name__ == "__main__":
    config_path = determine_config_path(DEFAULT_CONFIG_PATH)
    zeroshot_cfg = load_chronos_supervised_config(config_path)
    chronos_cfg = tu.load_config(zeroshot_cfg.training_config_path)

    seed = zeroshot_cfg.seed if zeroshot_cfg.seed is not None else chronos_cfg.seed
    tu.set_seed(seed)
    torch.manual_seed(seed)

    device = default_device()
    print(f"Using device: {device}")
    print(f"Requested horizons: {zeroshot_cfg.horizons}")
    print(f"Using zero-shot configuration: {zeroshot_cfg.config_path}")
    print(f"Using Chronos supervised config: {zeroshot_cfg.training_config_path}")

    results_dir = zeroshot_cfg.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint_info, eval_horizons, max_horizon = build_chronos_model(
        checkpoint_path=zeroshot_cfg.checkpoint_path,
        training_config=chronos_cfg,
        requested_horizons=zeroshot_cfg.horizons,
        encoder_type=zeroshot_cfg.encoder_type,
        device=device,
    )

    checkpoint_timestamp = extract_checkpoint_timestamp(zeroshot_cfg.checkpoint_path)
    if checkpoint_timestamp is None:
        checkpoint_timestamp = Path(zeroshot_cfg.checkpoint_path).stem

    predictions_dir = results_dir / f"{zeroshot_cfg.output_prefix}_{checkpoint_timestamp}_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    module = TimeSeriesDataModule(
        dataset_name=zeroshot_cfg.dataset_name or "",
        dataset_names=zeroshot_cfg.dataset_names,
        data_dir=zeroshot_cfg.data_dir,
        batch_size=zeroshot_cfg.batch_size,
        val_batch_size=zeroshot_cfg.val_batch_size,
        num_workers=zeroshot_cfg.num_workers,
        pin_memory=True,
        normalize=True,
        filename=zeroshot_cfg.filename,
        train=True,
        val=True,
        test=True,
    )
    dataset_groups = module.get_dataloaders()
    if not dataset_groups:
        raise RuntimeError("No datasets available for evaluation. Check filters or data directory.")

    results_by_dataset: Dict[str, Dict[int, Dict[str, float]]] = {}
    prediction_payloads: Dict[str, Dict[str, object]] = {}
    for group in dataset_groups:
        loader, dataset_obj, split_used = select_loader(group, zeroshot_cfg.split)
        if loader is None:
            print(f"Skipping dataset '{group.name}' because neither val nor test splits are available.")
            continue

        try:
            ensure_dataloader_pred_len(loader, max_horizon)
            print(f"\nEvaluating dataset '{group.name}' on '{split_used}' split...")
            metrics, payload = evaluate_and_collect(
                model,
                loader,
                device,
                list(eval_horizons),
                max_horizon,
                sequence_first_input=True,
                dataset=dataset_obj,
                dataset_name=group.name,
            )
            if metrics is None or payload is None:
                print(f"  No metrics computed for dataset '{group.name}'.")
                continue

            results_by_dataset[group.name] = metrics
            payload["dataset"] = group.name
            payload["dataset_slug"] = dataset_slug(group.name)
            payload["split"] = split_used
            prediction_payloads[group.name] = payload

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
                print(f"  Skipping dataset '{group.name}' due to out of memory error: {exc}")
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
        prefix=zeroshot_cfg.output_prefix,
        timestamp=checkpoint_timestamp,
    )

    print_evaluation_summary(results_by_dataset, checkpoint_info)

    horizon_summary = aggregate_results_by_horizon(results_by_dataset)
    horizon_json_path, horizon_csv_path = save_horizon_summary(
        horizon_summary,
        results_dir=results_dir,
        prefix=zeroshot_cfg.output_prefix,
        timestamp=checkpoint_timestamp,
    )

    print("\nSaved evaluation artifacts:")
    print(f"  Per-dataset JSON: {dataset_json_path}")
    print(f"  Per-dataset CSV:  {dataset_csv_path}")
    print(f"  Per-horizon JSON: {horizon_json_path}")
    print(f"  Per-horizon CSV:  {horizon_csv_path}")
    print(f"  Predictions dir:  {predictions_dir}")

    print("\nModel artifact references:")
    print(f"  Training config: {zeroshot_cfg.training_config_path}")
    print(f"  Chronos checkpoint: {zeroshot_cfg.checkpoint_path}")

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
