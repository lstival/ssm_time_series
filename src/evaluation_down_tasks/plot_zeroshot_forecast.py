"""Plot zero-shot forecast results saved by ICML_zeroshot_forecast."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter
from evaluation_down_tasks.zeroshot_utils import (
    PlotConfig,
    dataset_slug,
    determine_config_path,
    load_plot_config,
)

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


CONFIG_ENV_VAR = "ICML_ZEROSHOT_PLOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "icml_zeroshot_plot.yaml"
PLOT_HORIZONS: Tuple[int, ...] = (96, 192, 336, 720)
TOP_K = 3


def _extract_sample(payload: Dict[str, object], sample_idx: int) -> Dict[str, torch.Tensor]:
    sample = {
        "context": payload["context"][sample_idx],
        "targets": payload["targets"][sample_idx],
        "predictions": payload["predictions"][sample_idx],
    }
    if "context_normalized" in payload:
        sample["context_normalized"] = payload["context_normalized"][sample_idx]
        sample["targets_normalized"] = payload["targets_normalized"][sample_idx]
        sample["predictions_normalized"] = payload["predictions_normalized"][sample_idx]
    return sample


def _compute_sample_mae(
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    if not isinstance(targets, torch.Tensor) or not isinstance(predictions, torch.Tensor):
        raise TypeError("'targets' and 'predictions' must be torch.Tensor instances")
    if targets.ndim != 3 or predictions.ndim != 3:
        raise ValueError("Expected three-dimensional tensors: [samples, horizon, features]")

    common_horizon = int(min(targets.size(1), predictions.size(1)))
    if common_horizon <= 0:
        raise ValueError("Targets and predictions must contain at least one forecast step")

    targets_slice = targets[:, :common_horizon, :].float()
    predictions_slice = predictions[:, :common_horizon, :].float()
    mae_per_sample = torch.mean(torch.abs(targets_slice - predictions_slice), dim=(1, 2))
    return mae_per_sample, common_horizon


def _checkpoint_slug(predictions_path: Path, payload: Dict[str, object]) -> str:
    raw_name = (
        payload.get("checkpoint_name")
        or payload.get("checkpoint")
        or payload.get("checkpoint_path")
        or predictions_path.stem
    )
    if isinstance(raw_name, (str, Path)):
        raw_str = str(raw_name)
    else:
        raw_str = predictions_path.stem
    candidate = Path(raw_str).stem
    return dataset_slug(candidate)


def _extract_results_folder_name(predictions_path: Path) -> str:
    """Extract the results folder name from the predictions file path.
    
    For a path like /path/to/results/model_run_20241121/dataset_name/data.pt,
    this returns 'model_run_20241121'.
    """
    # Navigate up from data.pt -> dataset_dir -> results_folder
    # predictions_path is typically: results_folder/dataset_name/data.pt
    dataset_dir = predictions_path.parent  # dataset_name folder
    results_folder = dataset_dir.parent    # results folder (what we want)
    return results_folder.name


def _plot_single(
    *,
    context: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    horizon: int,
    feature_idx: int,
    dataset: str,
    sample_idx: int,
    output_dir: Path,
    show: bool,
    dpi: int,
    figsize: Sequence[float],
    context_denorm: Optional[torch.Tensor] = None,
) -> Path:
    context_np = context.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()

    if horizon > targets_np.shape[0]:
        raise ValueError(f"Requested horizon {horizon} exceeds stored targets length {targets_np.shape[0]}")

    if feature_idx >= context_np.shape[1]:
        raise ValueError(
            f"Feature index {feature_idx} out of range for context features {context_np.shape[1]}"
        )

    history = context_np[:, feature_idx]
    gt = targets_np[:horizon, feature_idx]
    pred = predictions_np[:horizon, feature_idx]

    history_len = history.shape[0]
    x_history = range(history_len)
    x_future = range(history_len, history_len + horizon)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_history, history, label="Context (input)", color="#1f77b4")
    ax.plot(x_future, gt, label="Ground Truth", color="#2ca02c")
    ax.plot(x_future, pred, label="Prediction", color="#d62728", linestyle="--")
    ax.axvline(history_len - 0.5, color="gray", linestyle=":", alpha=0.6)

    # If denormalized context is available, adjust Y-axis labels to show original values
    if context_denorm is not None:
        c_norm = context_np[:, feature_idx]
        c_denorm = context_denorm.detach().cpu().numpy()[:, feature_idx]
        
        # Determine linear mapping: denorm = a * norm + b
        norm_min, norm_max = c_norm.min(), c_norm.max()
        denorm_min, denorm_max = c_denorm.min(), c_denorm.max()
        
        if abs(norm_max - norm_min) > 1e-7:
            a = (denorm_max - denorm_min) / (norm_max - norm_min)
            b = denorm_min - a * norm_min
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{a * x + b:.2f}"))
        else:
            # If input is flat, we can't infer slope but we can at least show the level
            level = c_denorm[0]
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{level:.2f}"))

    ax.set_title(f"{dataset} | H{horizon} | Feature {feature_idx}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_slug(dataset)}_sample{sample_idx}_H{horizon}_feat{feature_idx}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


if __name__ == "__main__":
    config_path = determine_config_path(DEFAULT_CONFIG_PATH)
    plot_cfg = load_plot_config(config_path)

    generated_paths: List[Path] = []

    for dataset_name in plot_cfg.dataset_names:
        predictions_path = plot_cfg.dataset_files[dataset_name]
        payload: Dict[str, object] = torch.load(predictions_path, map_location="cpu")
        available_horizons = payload.get("eval_horizons") or []
        max_horizon = int(payload.get("max_horizon", 0))
        dataset_label = payload.get("dataset") or dataset_name
        results_folder_name = _extract_results_folder_name(predictions_path)
        base_output_dir = plot_cfg.output_dir / results_folder_name
        plots_output_dir = base_output_dir / "plots"
        best_metrics_root = base_output_dir / "best_metrics"
        dataset_best_dir = best_metrics_root / dataset_slug(str(dataset_label))

        targets_tensor = payload.get("targets")
        predictions_tensor = payload.get("predictions")
        if not isinstance(targets_tensor, torch.Tensor) or not isinstance(predictions_tensor, torch.Tensor):
            raise TypeError("Payload must contain 'targets' and 'predictions' tensors")
        mae_per_sample, comparable_horizon = _compute_sample_mae(
            targets_tensor,
            predictions_tensor,
        )
        num_samples = int(mae_per_sample.numel())
        top_k = min(TOP_K, num_samples)
        if top_k == 0:
            print("No samples available to rank for plotting; skipping dataset.")
            continue

        ranked_indices = torch.argsort(mae_per_sample)[:top_k].tolist()
        print(f"\nSelected top {top_k} sample(s) by MAE for dataset '{dataset_label}':")
        for rank, sample_idx in enumerate(ranked_indices, start=1):
            mae_value = float(mae_per_sample[sample_idx])
            print(f"  #{rank}: sample {sample_idx} | MAE={mae_value:.6f}")

        print(f"\nLoaded predictions from: {predictions_path}")
        print(f"Dataset: {dataset_label}")
        print(f"Available horizons: {available_horizons}")
        print(f"Context length: {payload.get('context_length')} | Target features: {payload.get('target_features')}")

        requested_horizons = [h for h in PLOT_HORIZONS if h <= comparable_horizon]
        missing_horizons = [h for h in PLOT_HORIZONS if h > comparable_horizon]
        if missing_horizons:
            print(
                "Warning: skipping horizons exceeding available forecast length "
                f"({comparable_horizon}). Missing: {missing_horizons}"
            )
        if not requested_horizons:
            print("Requested plot horizons are unavailable for this dataset; skipping plots.")
            continue

        for sample_idx in ranked_indices:
            sample = _extract_sample(payload, sample_idx)
            sample_target_len = sample["targets"].shape[0]
            sample_pred_len = sample["predictions"].shape[0]
            max_sample_horizon = min(sample_target_len, sample_pred_len)

            for horizon in requested_horizons:
                if max_horizon and horizon > max_horizon:
                    print(
                        f"Skipping horizon {horizon} for sample {sample_idx}: exceeds stored max horizon {max_horizon}."
                    )
                    continue
                if horizon > max_sample_horizon:
                    print(
                        f"Skipping horizon {horizon} for sample {sample_idx}: available forecast length {max_sample_horizon}."
                    )
                    continue
                for feature_idx in plot_cfg.feature_indices:
                    if feature_idx >= sample["context"].shape[1]:
                        print(
                            f"Skipping feature index {feature_idx} for sample {sample_idx}: "
                            f"only {sample['context'].shape[1]} feature(s) available."
                        )
                        continue
                    
                    # If normalized data is available, plot it but label axis with denorm data
                    if "context_normalized" in sample:
                        p_context = sample["context_normalized"]
                        p_targets = sample["targets_normalized"]
                        p_predictions = sample["predictions_normalized"]
                        p_context_denorm = sample["context"]
                    else:
                        p_context = sample["context"]
                        p_targets = sample["targets"]
                        p_predictions = sample["predictions"]
                        p_context_denorm = None

                    output_path = _plot_single(
                        context=p_context,
                        targets=p_targets,
                        predictions=p_predictions,
                        horizon=horizon,
                        feature_idx=feature_idx,
                        dataset=str(dataset_label),
                        sample_idx=sample_idx,
                        output_dir=plots_output_dir,
                        show=plot_cfg.show,
                        dpi=plot_cfg.dpi,
                        figsize=plot_cfg.figsize,
                        context_denorm=p_context_denorm,
                    )
                    dataset_best_dir.mkdir(parents=True, exist_ok=True)
                    best_output_path = dataset_best_dir / output_path.name
                    shutil.copy2(output_path, best_output_path)
                    generated_paths.append(output_path)
                    print(f"Saved plot: {output_path}")
                    print(f"Saved best plot copy: {best_output_path}")

    print(f"\nGenerated {len(generated_paths)} plot(s).")
