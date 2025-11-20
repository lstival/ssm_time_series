"""Plot zero-shot forecast results saved by ICML_zeroshot_forecast."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


CONFIG_ENV_VAR = "ICML_ZEROSHOT_PLOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "icml_zeroshot_plot.yaml"


def _dataset_slug(name: str) -> str:
    slug = name.replace("\\", "__").replace("/", "__").strip()
    return slug or "dataset"


@dataclass
class PlotConfig:
    config_path: Path
    predictions_root: Path
    predictions_file: Path
    dataset: str
    horizon_list: List[int]
    sample_indices: List[int]
    feature_indices: List[int]
    output_dir: Path
    show: bool
    dpi: int
    figsize: Sequence[float]


DEFAULT_CONFIG_CONTENT = {
    "predictions": {
        "root_dir": "../../results/icml_zeroshot_forecast_20251119_161152_predictions",
        "dataset": "dataset",
        "file": None,
    },
    "plot": {
        "horizons": [96],
        "sample_indices": [0],
        "feature_indices": [0],
        "output_dir": "../../results/plots",
        "show": False,
        "dpi": 150,
        "figsize": [12, 6],
    },
}


def _determine_config_path() -> Path:
    candidate: Optional[Path] = None
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
    elif os.getenv(CONFIG_ENV_VAR):
        candidate = Path(os.environ[CONFIG_ENV_VAR])
    else:
        candidate = DEFAULT_CONFIG_PATH

    if candidate == DEFAULT_CONFIG_PATH and not candidate.exists():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        with candidate.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(DEFAULT_CONFIG_CONTENT, handle, sort_keys=False)

    resolved = candidate.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Plot configuration file not found: {candidate}")
    return resolved


def _ensure_sequence(value: Optional[Iterable[int]], *, default: List[int]) -> List[int]:
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
    if root_dir_value is None and predictions_section.get("file") is None:
        raise ValueError("Configuration must provide either 'predictions.root_dir' or 'predictions.file'.")

    dataset_name = str(predictions_section.get("dataset") or "dataset")
    dataset_slug = _dataset_slug(dataset_name)

    if predictions_section.get("file") is not None:
        predictions_file = Path(predictions_section["file"]).expanduser().resolve()
    else:
        predictions_root = Path(root_dir_value).expanduser().resolve()
        predictions_file = predictions_root / dataset_slug / "data.pt"

    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")

    predictions_root = predictions_file.parent.parent

    plot_section = dict(payload.get("plot") or {})
    horizons = _ensure_sequence(plot_section.get("horizons"), default=[96])
    sample_indices = _ensure_sequence(plot_section.get("sample_indices"), default=[0])
    feature_indices = _ensure_sequence(plot_section.get("feature_indices"), default=[0])

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
        predictions_file=predictions_file,
        dataset=dataset_name,
        horizon_list=horizons,
        sample_indices=sample_indices,
        feature_indices=feature_indices,
        output_dir=output_dir,
        show=show,
        dpi=dpi,
        figsize=figsize,
    )


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
    output_path = output_dir / f"{_dataset_slug(dataset)}_sample{sample_idx}_H{horizon}_feat{feature_idx}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


if __name__ == "__main__":
    config_path = _determine_config_path()
    plot_cfg = load_plot_config(config_path)

    payload: Dict[str, object] = torch.load(plot_cfg.predictions_file, map_location="cpu")
    available_horizons = payload.get("eval_horizons") or []
    max_horizon = int(payload.get("max_horizon", 0))
    dataset_name = payload.get("dataset") or plot_cfg.dataset

    print(f"Loaded predictions from: {plot_cfg.predictions_file}")
    print(f"Dataset: {dataset_name}")
    print(f"Available horizons: {available_horizons}")
    print(f"Context length: {payload.get('context_length')} | Target features: {payload.get('target_features')}")

    generated_paths: List[Path] = []
    for sample_idx in plot_cfg.sample_indices:
        sample = _extract_sample(payload, sample_idx)
        for horizon in plot_cfg.horizon_list:
            if horizon > max_horizon:
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
                    dataset=str(dataset_name),
                    sample_idx=sample_idx,
                    output_dir=plot_cfg.output_dir,
                    show=plot_cfg.show,
                    dpi=plot_cfg.dpi,
                    figsize=plot_cfg.figsize,
                )
                generated_paths.append(output_path)
                print(f"Saved plot: {output_path}")

    print(f"Generated {len(generated_paths)} plot(s).")
