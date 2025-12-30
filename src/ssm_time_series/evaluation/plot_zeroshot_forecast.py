"""Plot zero-shot forecast results saved by ICML_zeroshot_forecast."""

from __future__ import annotations

import shutil
# Removed legacy sys.path hack


CONFIG_ENV_VAR = "ICML_ZEROSHOT_PLOT_CONFIG"
DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "icml_zeroshot_plot.yaml"
PLOT_HORIZONS: Tuple[int, ...] = (96, 192, 336, 720)
BEST_K = 3
MAX_PLOTS_PER_DATASET = 100


def _extract_sample(payload: Dict[str, object], sample_idx: int) -> Dict[str, torch.Tensor]:
    context = payload["context"][sample_idx]
    targets = payload["targets"][sample_idx]
    preds = payload["predictions"][sample_idx]
    return {
        "context": context,
        "targets": targets,
        "predictions": preds,
    }


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
        if num_samples == 0:
            print("No samples available to rank for plotting; skipping dataset.")
            continue

        ranked_indices = torch.argsort(mae_per_sample).tolist()
        best_k = min(BEST_K, num_samples)
        best_indices = ranked_indices[:best_k]
        print(f"\nSelected best {best_k} sample(s) by MAE for dataset '{dataset_label}':")
        for rank, sample_idx in enumerate(best_indices, start=1):
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

        plots_budget = int(MAX_PLOTS_PER_DATASET)
        if plots_budget <= 0:
            print("Plot budget is non-positive; skipping plots.")
            continue

        num_features = len(plot_cfg.feature_indices)
        if num_features == 0:
            print("No feature indices provided in config; skipping plots.")
            continue

        # Estimate how many samples we need to hit the budget.
        plots_per_sample_estimate = max(1, len(requested_horizons) * num_features)
        samples_needed = int((plots_budget + plots_per_sample_estimate - 1) // plots_per_sample_estimate)
        samples_needed = max(samples_needed, best_k)
        plot_sample_indices = ranked_indices[: min(num_samples, samples_needed)]
        print(
            f"Will generate up to {plots_budget} plot(s) for dataset '{dataset_label}' "
            f"from {len(plot_sample_indices)} sample(s)."
        )

        plots_generated_for_dataset = 0

        for sample_idx in plot_sample_indices:
            if plots_generated_for_dataset >= plots_budget:
                break
            sample = _extract_sample(payload, sample_idx)
            sample_target_len = sample["targets"].shape[0]
            sample_pred_len = sample["predictions"].shape[0]
            max_sample_horizon = min(sample_target_len, sample_pred_len)

            for horizon in requested_horizons:
                if plots_generated_for_dataset >= plots_budget:
                    break
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
                    if plots_generated_for_dataset >= plots_budget:
                        break
                    if feature_idx >= sample["context"].shape[1]:
                        print(
                            f"Skipping feature index {feature_idx} for sample {sample_idx}: "
                            f"only {sample['context'].shape[1]} feature(s) available."
                        )
                        continue
                    output_path = _plot_single(
                        context=sample["context"],
                        targets=sample["targets"],
                        predictions=sample["predictions"],
                        horizon=horizon,
                        feature_idx=feature_idx,
                        dataset=str(dataset_label),
                        sample_idx=sample_idx,
                        output_dir=plots_output_dir,
                        show=plot_cfg.show,
                        dpi=plot_cfg.dpi,
                        figsize=plot_cfg.figsize,
                    )
                    generated_paths.append(output_path)
                    plots_generated_for_dataset += 1
                    print(f"Saved plot: {output_path}")

                    if sample_idx in best_indices:
                        dataset_best_dir.mkdir(parents=True, exist_ok=True)
                        best_output_path = dataset_best_dir / output_path.name
                        shutil.copy2(output_path, best_output_path)
                        print(f"Saved best plot copy: {best_output_path}")

        if plots_generated_for_dataset < plots_budget:
            print(
                f"Generated {plots_generated_for_dataset} plot(s) for dataset '{dataset_label}' "
                f"(budget was {plots_budget})."
            )

    print(f"\nGenerated {len(generated_paths)} plot(s).")
