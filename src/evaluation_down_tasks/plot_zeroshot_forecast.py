"""Plot zero-shot forecast results saved by ICML_zeroshot_forecast."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
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




def _extract_sample(payload: Dict[str, object], sample_idx: int) -> Dict[str, torch.Tensor]:
    context = payload["context"][sample_idx]
    targets = payload["targets"][sample_idx]
    preds = payload["predictions"][sample_idx]
    return {
        "context": context,
        "targets": targets,
        "predictions": preds,
    }


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

    ax.set_title(f"{dataset} | Sample {sample_idx} | H{horizon} | Feature {feature_idx}")
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

        print(f"\nLoaded predictions from: {predictions_path}")
        print(f"Dataset: {dataset_label}")
        print(f"Available horizons: {available_horizons}")
        print(f"Context length: {payload.get('context_length')} | Target features: {payload.get('target_features')}")

        for sample_idx in plot_cfg.sample_indices:
            sample = _extract_sample(payload, sample_idx)
            for horizon in plot_cfg.horizon_list:
                if max_horizon and horizon > max_horizon:
                    raise ValueError(
                        f"Requested horizon {horizon} exceeds stored maximum horizon {max_horizon}"
                    )
                for feature_idx in plot_cfg.feature_indices:
                    output_path = _plot_single(
                        context=sample["context"],
                        targets=sample["targets"],
                        predictions=sample["predictions"],
                        horizon=horizon,
                        feature_idx=feature_idx,
                        dataset=str(dataset_label),
                        sample_idx=sample_idx,
                        output_dir=plot_cfg.output_dir,
                        show=plot_cfg.show,
                        dpi=plot_cfg.dpi,
                        figsize=plot_cfg.figsize,
                    )
                    generated_paths.append(output_path)
                    print(f"Saved plot: {output_path}")

    print(f"\nGenerated {len(generated_paths)} plot(s).")
