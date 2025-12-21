"""Debug script: inspect Chronos *raw* series normalization and plotting.

What it does
- Reads the Chronos loader config referenced by `src/configs/chronos_supervised.yaml` (data.cronos_config)
- Loads ONE dataset at a time (no concatenation)
- Uses Chronos *raw* series (no patching, no padding, no interpolation)
- Computes global normalization params across the full split (train by default)
- Plots (context, gt) at the original series length:
    - normalized scale
    - inverse-normalized (should match the raw series)

Usage (PowerShell)
- `python src/debug_chronos_normalization.py`
- `python src/debug_chronos_normalization.py --dataset m4_yearly --examples 3`

Outputs are written under `checkpoints/debug_chronos_normalization/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml

import training_utils as tu
from moco_training import resolve_path


def _find(cfg, key: str):
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        for v in cfg.values():
            found = _find(v, key)
            if found is not None:
                return found
    if isinstance(cfg, list):
        for item in cfg:
            found = _find(item, key)
            if found is not None:
                return found
    return None


def _group_dataframe(df) -> Iterable[Tuple[object, "np.ndarray"]]:
    # Mirrors the logic used elsewhere in the repo.
    if getattr(df.index, "duplicated", None) is not None and df.index.duplicated().any():
        grouped = df.groupby(level=0, sort=False)
        for key, frame in grouped:
            yield key, frame
        return

    group_col = None
    for col in ("item_id", "series", "segment"):
        if col in df.columns:
            group_col = col
            break
    if group_col is None:
        raise ValueError("Could not infer series groups (expected item_id/series/segment or unique index).")

    grouped = df.groupby(group_col, sort=False)
    for key, frame in grouped:
        yield key, frame


def _compute_raw_series_stats(
    *,
    config_path: Path,
    data_cfg: dict,
    dataset_name: str,
) -> Dict[object, Tuple[float, float, float]]:
    """Return stats keyed by original series id: {series_key: (min, max, std)}."""

    from dataloaders.utils import ensure_hf_list_feature_registered
    from dataloaders.cronos_dataset import load_chronos_datasets, to_pandas

    ensure_hf_list_feature_registered()

    split_name = str(data_cfg.get("split") or "train")

    # Reuse loader defaults from prepare_dataset.
    load_kwargs = dict(data_cfg.get("load_kwargs", {}) or {})
    data_dir = config_path.parent.parent / "data"
    load_kwargs.setdefault("offline_cache_dir", str(data_dir))
    load_kwargs.setdefault("force_offline", True)

    hf_raw = load_chronos_datasets(
        [dataset_name],
        split=split_name,
        normalize_per_series=False,
        **load_kwargs,
    )
    df = to_pandas(hf_raw)

    stats: Dict[object, Tuple[float, float, float]] = {}
    for key, frame in _group_dataframe(df):
        values = np.asarray(frame["target"].to_numpy(), dtype=np.float64)
        if values.size == 0:
            stats[key] = (0.0, 0.0, 0.0)
            continue
        stats[key] = (
            float(np.nanmin(values)),
            float(np.nanmax(values)),
            float(np.nanstd(values)),
        )

    return stats


def _infer_normalize_mode(raw_cronos_cfg: dict) -> str:
    mode = _find(raw_cronos_cfg, "normalize_mode")
    if isinstance(mode, str) and mode.strip():
        parsed = mode.strip().lower()
        # Force min-max normalization for debugging consistency with supervised training.
        # Accept common aliases.
        if parsed in ("global_standard", "standard", "zscore"):
            return "global_minmax"
        return parsed

    normalize = _find(raw_cronos_cfg, "normalize")
    if isinstance(normalize, bool):
        return "global_minmax" if normalize else "none"

    return "none"


def _compute_global_norm_params(values: np.ndarray, *, mode: str, epsilon: float = 1e-12) -> Optional[Dict[str, Any]]:
    mode = str(mode or "none").lower()
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or mode in ("none", "false", "0"):
        return None

    if mode in ("global_minmax", "minmax"):
        dmin = float(np.min(values))
        dmax = float(np.max(values))
        return {"mode": "global_minmax", "min": dmin, "max": dmax, "epsilon": float(epsilon)}

    if mode in ("global_standard", "standard", "zscore"):
        mean = float(np.mean(values))
        std = float(np.std(values))
        return {"mode": "global_standard", "mean": mean, "std": std, "epsilon": float(epsilon)}

    return None


def _normalize_array(values: np.ndarray, params: Optional[Dict[str, Any]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if params is None:
        return values
    mode = str(params.get("mode") or "").lower()
    eps = float(params.get("epsilon", 1e-12))
    if mode == "global_minmax":
        dmin = float(params.get("min", 0.0))
        dmax = float(params.get("max", 0.0))
        rng = dmax - dmin
        denom = rng if abs(rng) > eps else 1.0
        return (values - dmin) / denom
    if mode == "global_standard":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        denom = std if abs(std) > eps else 1.0
        return (values - mean) / denom
    return values


def _denormalize_array(values: np.ndarray, params: Optional[Dict[str, Any]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if params is None:
        return values
    mode = str(params.get("mode") or "").lower()
    eps = float(params.get("epsilon", 1e-12))
    if mode == "global_minmax":
        dmin = float(params.get("min", 0.0))
        dmax = float(params.get("max", 0.0))
        rng = dmax - dmin
        denom = rng if abs(rng) > eps else 1.0
        return values * denom + dmin
    if mode == "global_standard":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        denom = std if abs(std) > eps else 1.0
        return values * denom + mean
    return values


def _plot_context_gt(
    context: np.ndarray,
    target: np.ndarray,
    *,
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ctx = np.asarray(context).reshape(-1)
    tgt = np.asarray(target).reshape(-1)

    x_ctx = np.arange(ctx.size)
    x_tgt = np.arange(ctx.size, ctx.size + tgt.size)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(x_ctx, ctx, label="input", linewidth=1.5)
    ax.plot(x_tgt, tgt, label="gt", linewidth=1.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "chronos_supervised.yaml"),
        help="Path to chronos_supervised.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to inspect (defaults to first in cronos loader config)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="How many steps to treat as gt (defaults to ~1/3 of series length)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="How many steps to treat as input (defaults to remainder)",
    )
    parser.add_argument("--examples", type=int, default=3, help="How many windows to plot")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path("checkpoints") / "debug_chronos_normalization"),
        help="Output directory for plots",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = tu.load_config(config_path)
    tu.set_seed(cfg.seed)

    data_cfg = cfg.data
    training_cfg = cfg.training

    cronos_config = data_cfg.get("cronos_config")
    if cronos_config is None:
        cronos_config = config_path.parent / "cronos_loader_example.yaml"
    cronos_config = resolve_path(config_path.parent, cronos_config)
    if cronos_config is None or not cronos_config.exists():
        raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

    with open(cronos_config, "r", encoding="utf-8") as handle:
        raw_cronos_cfg = yaml.safe_load(handle) or {}

    dataset_names = _find(raw_cronos_cfg, "datasets_to_load")
    if not isinstance(dataset_names, Sequence) or isinstance(dataset_names, (str, bytes)) or not dataset_names:
        raise ValueError(f"No datasets_to_load found in {cronos_config}")

    dataset_name = str(args.dataset) if args.dataset is not None else str(dataset_names[0])
    print(f"Dataset: {dataset_name}")

    # Compute raw per-series stats (min/max/std on full series; no normalization).
    per_dataset_cfg = dict(data_cfg)
    per_dataset_cfg["dataset_name"] = dataset_name
    raw_stats_by_key = _compute_raw_series_stats(config_path=config_path, data_cfg=per_dataset_cfg, dataset_name=dataset_name)
    print(f"Raw series stats available for {len(raw_stats_by_key)} series keys.")
    if raw_stats_by_key:
        mins = np.asarray([v[0] for v in raw_stats_by_key.values()], dtype=np.float64)
        maxs = np.asarray([v[1] for v in raw_stats_by_key.values()], dtype=np.float64)
        stds = np.asarray([v[2] for v in raw_stats_by_key.values()], dtype=np.float64)
        print(
            "Raw stats summary: "
            f"min(min)={np.nanmin(mins):.6g} | max(max)={np.nanmax(maxs):.6g} | "
            f"mean(std)={np.nanmean(stds):.6g}"
        )

    # Load the raw dataframe once (we'll use it for plotting).
    from dataloaders.utils import ensure_hf_list_feature_registered
    from dataloaders.cronos_dataset import load_chronos_datasets, to_pandas

    ensure_hf_list_feature_registered()

    split_name = str(per_dataset_cfg.get("split") or "train")
    load_kwargs = dict(per_dataset_cfg.get("load_kwargs", {}) or {})
    data_dir = config_path.parent.parent / "data"
    load_kwargs.setdefault("offline_cache_dir", str(data_dir))
    load_kwargs.setdefault("force_offline", True)

    hf_raw = load_chronos_datasets(
        [dataset_name],
        split=split_name,
        normalize_per_series=False,
        **load_kwargs,
    )
    df = to_pandas(hf_raw)

    # Global normalization params (computed over the entire split).
    normalize_mode = _infer_normalize_mode(raw_cronos_cfg)
    all_values = np.asarray(df["target"].to_numpy(), dtype=np.float64)
    norm_params = _compute_global_norm_params(all_values, mode=normalize_mode)

    if norm_params is None:
        print("Global normalization params: N/A (mode=none)")
    else:
        mode = str(norm_params.get("mode"))
        if mode == "global_minmax":
            print(
                "Global normalization params: "
                f"mode={mode} min={norm_params.get('min'):.6g} max={norm_params.get('max'):.6g}"
            )
        elif mode == "global_standard":
            print(
                "Global normalization params: "
                f"mode={mode} mean={norm_params.get('mean'):.6g} std={norm_params.get('std'):.6g}"
            )

    outdir = Path(args.outdir).expanduser().resolve() / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Save normalization params for reproducible inverse-transform.
    if norm_params is not None:
        with open(outdir / "normalization.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(norm_params, handle, sort_keys=True)

    # Plot first N series keys (original length).
    series_keys = list(raw_stats_by_key.keys())
    n = min(int(args.examples), len(series_keys))
    if n <= 0:
        raise ValueError(f"No series found for dataset '{dataset_name}'.")

    for i in range(n):
        series_key = series_keys[i]
        frame = next(frame for key, frame in _group_dataframe(df) if key == series_key)
        raw_values = np.asarray(frame["target"].to_numpy(), dtype=np.float64).reshape(-1)
        raw_values = raw_values[np.isfinite(raw_values)]
        if raw_values.size < 2:
            print(f"example {i}: series_key={series_key} | too short, skipped")
            continue

        # Choose context/gt split without changing the total length.
        if args.context_len is None and args.horizon is None:
            horizon = max(1, raw_values.size // 3)
            context_len = raw_values.size - horizon
        else:
            horizon = int(args.horizon) if args.horizon is not None else max(1, raw_values.size // 3)
            horizon = max(1, min(horizon, raw_values.size - 1))

            if args.context_len is None:
                context_len = raw_values.size - horizon
            else:
                context_len = int(args.context_len)
                context_len = max(1, min(context_len, raw_values.size - horizon))

            if context_len + horizon > raw_values.size:
                horizon = max(1, raw_values.size - context_len)

        print(f"example {i}: series_key={series_key} | len={raw_values.size} | context_len={context_len} | horizon={horizon}")

        ctx_raw = raw_values[:context_len]
        tgt_raw = raw_values[context_len:context_len + horizon]

        ctx_norm = _normalize_array(ctx_raw, norm_params)
        tgt_norm = _normalize_array(tgt_raw, norm_params)

        ctx_denorm = _denormalize_array(ctx_norm, norm_params)
        tgt_denorm = _denormalize_array(tgt_norm, norm_params)

        _plot_context_gt(
            ctx_norm,
            tgt_norm,
            title=f"normalized | dataset={dataset_name} | series_key={series_key} | example={i}",
            save_path=outdir / f"example_{i:02d}_normalized.png",
        )
        _plot_context_gt(
            ctx_denorm,
            tgt_denorm,
            title=f"denormalized | dataset={dataset_name} | series_key={series_key} | example={i}",
            save_path=outdir / f"example_{i:02d}_denormalized.png",
        )

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
