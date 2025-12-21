"""Train a single encoder for Chronos forecasting using YAML-configured datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import comet_ml
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import training_utils as tu
import util as u
from moco_training import (
    build_dataloaders,
    prepare_dataset,
    resolve_checkpoint_dir,
    resolve_path,
    split_dataset,
)


def _denormalize(
    tensor: torch.Tensor,
    series_ids: Optional[torch.Tensor],
    denorm: Optional[Any],
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Inverse-transform predictions/targets for metrics/plots.

    Supports:
    - Dataset-level global normalization: dict with {'mode': 'global_minmax'|'global_standard', ...}
    - Legacy per-series min-max stats: dict[int, (min, max)]
    """
    if denorm is None:
        return tensor
    if tensor.ndim != 3:
        return tensor

    # Dataset-level normalization (preferred)
    if isinstance(denorm, dict) and "mode" in denorm:
        mode = str(denorm.get("mode") or "").lower()
        if mode == "global_minmax":
            dmin = float(denorm.get("min", 0.0))
            dmax = float(denorm.get("max", 0.0))
            rng = dmax - dmin
            denom = rng if abs(rng) > float(denorm.get("epsilon", epsilon)) else 1.0
            return tensor * denom + dmin
        if mode == "global_standard":
            mean = float(denorm.get("mean", 0.0))
            std = float(denorm.get("std", 1.0))
            denom = std if abs(std) > float(denorm.get("epsilon", epsilon)) else 1.0
            return tensor * denom + mean
        return tensor

    # Legacy per-series min/max
    if series_ids is None or not isinstance(denorm, dict):
        return tensor

    series_ids_cpu = series_ids.detach().cpu().long().view(-1)
    out = tensor.detach().cpu().clone()
    for i, sid in enumerate(series_ids_cpu.tolist()):
        mm = denorm.get(int(sid))
        if mm is None:
            continue
        seq_min, seq_max = mm
        rng = float(seq_max) - float(seq_min)
        if abs(rng) < epsilon:
            out[i] = float(seq_min)
        else:
            out[i] = out[i] * rng + float(seq_min)
    return out.to(tensor.device)


def _save_normalization_params(path: Path, params: Optional[dict]) -> None:
    if params is None:
        return
    try:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(params, handle, sort_keys=True)
    except Exception:
        pass


def _write_cronos_minmax_override(
    cronos_config_path: Path,
    *,
    out_path: Path,
) -> Path:
    """Write a Chronos loader YAML that forces global min-max normalization.

    This keeps training behavior stable even if the referenced YAML uses
    `global_standard`.
    """
    import yaml

    cronos_config_path = Path(cronos_config_path)
    out_path = Path(out_path)

    with open(cronos_config_path, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    # Force global min-max.
    raw_cfg["normalize"] = True
    raw_cfg["normalize_mode"] = "global_minmax"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_cfg, handle, sort_keys=False)

    return out_path


def _plot_forecast_examples(
    context: torch.Tensor,
    target: torch.Tensor,
    preds: torch.Tensor,
    *,
    save_path: Path,
    max_examples: int = 5,
) -> None:
    import matplotlib.pyplot as plt

    ctx = context.detach().cpu().numpy()
    tgt = target.detach().cpu().numpy()
    prd = preds.detach().cpu().numpy()

    n = min(int(ctx.shape[0]), max_examples)
    if n <= 0:
        return

    # Plot first feature only to keep it readable.
    ctx_1 = ctx[..., 0]
    tgt_1 = tgt[..., 0]
    prd_1 = prd[..., 0]

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        x_ctx = np.arange(ctx_1.shape[1])
        x_fut = np.arange(ctx_1.shape[1], ctx_1.shape[1] + tgt_1.shape[1])
        ax.plot(x_ctx, ctx_1[i], label="input", linewidth=1.5)
        ax.plot(x_fut, tgt_1[i], label="gt", linewidth=1.5)
        ax.plot(x_fut, prd_1[i], label="pred", linewidth=1.5)
        ax.set_title(f"example {i}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


class ChronosForecastModel(nn.Module):
    """Simple forecasting head on top of a single encoder."""

    def __init__(self, encoder: nn.Module, input_features: int, target_dim: int, pred_len: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.pred_len = pred_len
        self.target_dim = target_dim

        encoder_channels = int(getattr(encoder, "input_dim", input_features))
        if input_features != encoder_channels:
            self.channel_adapter: nn.Module = nn.Conv1d(input_features, encoder_channels, kernel_size=1, bias=False)
        else:
            self.channel_adapter = nn.Identity()

        embedding_dim = int(getattr(encoder, "embedding_dim", encoder_channels))
        self.head = nn.Linear(embedding_dim, pred_len * target_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # Expect seq with shape (batch, time, features)
        x = seq.transpose(1, 2)  # -> (batch, features, time)
        x = self.channel_adapter(x)
        embedding = self.encoder(x)
        out = self.head(embedding)
        return out.view(seq.size(0), self.pred_len, self.target_dim)


def _prepare_forecast_batch(batch, pred_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
    seq_len = int(seq.size(1))
    if seq_len < 2:
        raise ValueError(f"Sequence length {seq_len} is too short for forecasting.")

    # Adaptive split for short series:
    # - Default: use configured pred_len (e.g. 96)
    # - If too short, use 50/50 split: pred_len := floor(seq_len/2)
    effective_pred_len = min(int(pred_len), seq_len // 2)
    effective_pred_len = max(1, effective_pred_len)

    context_len = seq_len - effective_pred_len
    if context_len < 1:
        context_len = 1
        effective_pred_len = max(1, seq_len - context_len)

    context = seq[:, :context_len, :]
    target = seq[:, context_len:context_len + effective_pred_len, :]
    return context, target, effective_pred_len


def train_one_epoch(
    model: ChronosForecastModel,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    pred_len: int,
    scaler: GradScaler,
    max_grad_norm: Optional[float],
    amp_enabled: bool,
) -> float:
    model.train()
    running = 0.0
    steps = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        context, target, effective_pred_len = _prepare_forecast_batch(batch, pred_len, device)
        optimizer.zero_grad(set_to_none=True)
        # AMP can underflow very small losses to 0 in float16. Keep forward in autocast,
        # but compute the loss in float32 outside autocast for numerical stability.
        with autocast(enabled=amp_enabled):  # mixed precision for model forward
            preds_full = model(context)
            preds = preds_full[:, :effective_pred_len, :]
        with autocast(enabled=False):
            loss = criterion(preds.float(), target.float())
        scaler.scale(loss).backward() # Use scale fator to avoid zero (gradient vanishing) in the update pass (because float16 can round the values to 0 when too small)
        if max_grad_norm is not None:
            scaler.unscale_(optimizer) # Remove the scale factor to avoid (inf and Nan) when clipping to a correct range
            clip_grad_norm_(model.parameters(), max_grad_norm) # Create a clip in the max and min in the gradient after the unscaling process
        scaler.step(optimizer) # Verify that gradient are in normal scalle e avoid Nan values
        scaler.update() # Update the gradient 
        running += loss.item()
        steps += 1
    return running / max(1, steps)


@torch.no_grad()
def evaluate(
    model: ChronosForecastModel,
    loader: Optional[DataLoader],
    criterion: nn.Module,
    *,
    device: torch.device,
    pred_len: int,
    denorm_stats: Optional[Any] = None,
    epoch: Optional[int] = None,
    plot_dir: Optional[Path] = None,
    experiment: Optional[comet_ml.Experiment] = None,
) -> Optional[Tuple[float, float]]:
    if loader is None:
        return None
    model.eval()
    mse_running = 0.0
    mae_running = 0.0
    steps = 0
    plotted = False
    for batch in tqdm(loader, desc="Val", leave=False):
        series_ids = None
        if isinstance(batch, (tuple, list)) and len(batch) > 1 and isinstance(batch[1], torch.Tensor):
            series_ids = batch[1].to(device)

        context, target, effective_pred_len = _prepare_forecast_batch(batch, pred_len, device)
        preds_full = model(context)
        preds = preds_full[:, :effective_pred_len, :]

        # Compute metrics on denormalized values (raw scale).
        denorm_context = _denormalize(context, series_ids, denorm_stats)
        denorm_target = _denormalize(target, series_ids, denorm_stats)
        denorm_preds = _denormalize(preds, series_ids, denorm_stats)

        diff = (denorm_preds.float() - denorm_target.float())
        mse = torch.mean(diff * diff).item()
        mae = torch.mean(torch.abs(diff)).item()
        mse_running += mse
        mae_running += mae
        steps += 1

        # Plot once per evaluation (first batch), denormalized.
        if (not plotted) and plot_dir is not None and epoch is not None:
            plot_path = Path(plot_dir) / f"val_examples_epoch_{epoch + 1:03d}.png"
            _plot_forecast_examples(denorm_context, denorm_target, denorm_preds, save_path=plot_path, max_examples=5)
            if experiment is not None:
                try:
                    experiment.log_image(str(plot_path), name=f"val_examples_epoch_{epoch + 1}")
                except Exception:
                    pass
            plotted = True

    mse_avg = mse_running / max(1, steps)
    mae_avg = mae_running / max(1, steps)

    if experiment is not None and epoch is not None:
        experiment.log_metric("val_mse", mse_avg, step=epoch + 1)
        experiment.log_metric("val_mae", mae_avg, step=epoch + 1)

    return mse_avg, mae_avg


def _train_encoder(
    model: ChronosForecastModel,
    *,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    training_cfg: dict,
    device: torch.device,
    epochs: int,
    pred_len: int,
    checkpoint_dir: Path,
    experiment: Optional[comet_ml.Experiment] = None,
) -> None:
    optimizer = tu.build_optimizer(model, training_cfg)
    scheduler = tu.build_scheduler(optimizer, training_cfg, epochs)
    criterion = nn.MSELoss()
    amp_enabled = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0)) or None
    scaler = GradScaler(enabled=amp_enabled)

    model.to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log hyperparameters to Comet
    if experiment is not None:
        experiment.log_parameters({
            "epochs": epochs,
            "pred_len": pred_len,
            "learning_rate": training_cfg.get("learning_rate", "N/A"),
            "weight_decay": training_cfg.get("weight_decay", 0.0),
            "batch_size": training_cfg.get("batch_size", "N/A"),
            "use_amp": amp_enabled,
            "max_grad_norm": max_grad_norm,
            "optimizer": training_cfg.get("optimizer", "AdamW"),
            "scheduler": training_cfg.get("scheduler", "None"),
            "device": str(device),
        })

    best_metric = float("inf")
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            pred_len=pred_len,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            amp_enabled=amp_enabled,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device=device,
            pred_len=pred_len,
            denorm_stats=training_cfg.get("denorm_stats"),
            epoch=epoch,
            plot_dir=checkpoint_dir / "plots",
            experiment=experiment,
        )

        val_loss = val_metrics[0] if val_metrics is not None else None

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss if val_loss is not None else train_loss)
        elif scheduler is not None:
            scheduler.step()

        # Log metrics to Comet
        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            experiment.log_metric("learning_rate", current_lr, step=epoch + 1)

        metric = val_loss if val_loss is not None else train_loss
        print(
            f"Epoch {epoch + 1}/{epochs} | train={train_loss:.6e}"
            + (
                f" | val_mse={val_metrics[0]:.6e} | val_mae={val_metrics[1]:.6e}"
                if val_metrics is not None
                else ""
            )
        )

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if metric < best_metric:
            best_metric = metric
            torch.save({**state, "best_metric": best_metric}, checkpoint_dir / "best.pt")


def _build_denorm_stats_for_dataset(
    *,
    config_path: Path,
    data_cfg: dict,
    dataset_name: str,
    dataset_obj,
) -> Optional[Dict[int, Tuple[float, float]]]:
    """Build per-series (min, max) stats for denormalization during validation.

    This uses the raw (non-normalized) Chronos split for the given dataset name.
    """
    try:
        # If the patched dataset already exposes normalization params, prefer those.
        base_dataset = getattr(dataset_obj, "dataset", dataset_obj)
        norm = getattr(base_dataset, "normalization", None)
        if isinstance(norm, dict) and "mode" in norm:
            return norm  # type: ignore[return-value]

        normalize = bool(data_cfg.get("normalize", True))
        if not normalize:
            return None

        from dataloaders.utils import ensure_hf_list_feature_registered
        from dataloaders.cronos_dataset import load_chronos_datasets, to_pandas

        ensure_hf_list_feature_registered()

        cronos_config = data_cfg.get("cronos_config")
        if cronos_config is None:
            cronos_config = config_path.parent / "cronos_loader_example.yaml"
        cronos_config = resolve_path(config_path.parent, cronos_config)
        if cronos_config is None or not Path(cronos_config).exists():
            raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

        # Reuse loader defaults from prepare_dataset.
        load_kwargs = dict(data_cfg.get("load_kwargs", {}) or {})
        data_dir = config_path.parent.parent / "data"
        load_kwargs.setdefault("offline_cache_dir", str(data_dir))
        load_kwargs.setdefault("force_offline", True)
        split_name = data_cfg.get("split") or "train"

        hf_raw = load_chronos_datasets(
            [dataset_name],
            split=str(split_name),
            normalize_per_series=False,
            **load_kwargs,
        )
        df = to_pandas(hf_raw)

        if df.index.duplicated().any():
            grouped = df.groupby(level=0, sort=False)
        else:
            group_col = None
            for col in ("item_id", "series", "segment"):
                if col in df.columns:
                    group_col = col
                    break
            if group_col is None:
                raise ValueError("Could not infer series groups for denormalization")
            grouped = df.groupby(group_col, sort=False)

        stats_by_key: Dict[object, Tuple[float, float]] = {}
        for key, frame in grouped:
            values = np.asarray(frame["target"].to_numpy(), dtype=np.float64)
            if values.size == 0:
                stats_by_key[key] = (0.0, 0.0)
            else:
                stats_by_key[key] = (float(np.nanmin(values)), float(np.nanmax(values)))

        lookup = getattr(base_dataset, "series_id_lookup", None)
        if not isinstance(lookup, dict):
            return None

        denorm_stats: Dict[int, Tuple[float, float]] = {}
        for numeric_id, series_key in lookup.items():
            mm = stats_by_key.get(series_key)
            if mm is not None:
                denorm_stats[int(numeric_id)] = mm

        return denorm_stats
    except Exception as exc:
        print(f"Warning: could not build denormalization stats for '{dataset_name}': {exc}")
        return None


def _train_encoder_sequential_datasets(
    model: ChronosForecastModel,
    *,
    config_path: Path,
    data_cfg: dict,
    training_cfg: dict,
    device: torch.device,
    pred_len: int,
    checkpoint_dir: Path,
    dataset_names: list[str],
    epochs_per_dataset: int,
    seed: int,
    experiment: Optional[comet_ml.Experiment] = None,
) -> None:
    """Train continuously, switching datasets every N epochs.

    The optimizer/scheduler/scaler state is preserved across dataset switches.
    """
    optimizer = tu.build_optimizer(model, training_cfg)
    scheduler = tu.build_scheduler(optimizer, training_cfg, epochs_per_dataset * max(1, len(dataset_names)))
    criterion = nn.MSELoss()
    amp_enabled = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"
    max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0)) or None
    scaler = GradScaler(enabled=amp_enabled)

    model.to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")
    global_epoch = 0
    for dataset_idx, dataset_name in enumerate(dataset_names):
        print(f"\n=== Dataset {dataset_idx + 1}/{len(dataset_names)}: {dataset_name} ===")
        if experiment is not None:
            experiment.log_parameter("current_dataset", dataset_name)

        # Load ONLY this dataset (no concatenation).
        per_dataset_cfg = dict(data_cfg)
        per_dataset_cfg["dataset_name"] = dataset_name
        # Do not override Chronos loader YAML patch_length (avoids accidental 1024).
        # Let the cronos_loader_example.yaml decide.
        per_dataset_cfg["patch_length"] = None
        dataset = prepare_dataset(config_path, per_dataset_cfg)

        # Persist normalization params for later inverse-transform.
        base_dataset = getattr(dataset, "dataset", dataset)
        norm_params = getattr(base_dataset, "normalization", None)
        if isinstance(norm_params, dict) and "mode" in norm_params:
            _save_normalization_params(checkpoint_dir / "normalization" / f"{dataset_name}.yaml", norm_params)

        val_ratio = float(per_dataset_cfg.get("val_ratio", 0.1))
        train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=seed)

        batch_size = int(per_dataset_cfg.get("batch_size", 128))
        num_workers = int(per_dataset_cfg.get("num_workers", 0))
        pin_memory = bool(per_dataset_cfg.get("pin_memory", False))

        train_loader, val_loader = build_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=batch_size,
            val_batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = val_loader if val_loader is not None and len(val_loader) > 0 else None

        # Build denorm stats for metrics/plots on this dataset.
        denorm_stats = _build_denorm_stats_for_dataset(
            config_path=config_path,
            data_cfg=per_dataset_cfg,
            dataset_name=dataset_name,
            dataset_obj=dataset,
        )

        # Validate pred_len compatibility on this dataset.
        sample_batch = next(iter(train_loader))
        sample_seq = u.prepare_sequence(u.extract_sequence(sample_batch))
        seq_len = sample_seq.shape[1]
        if int(seq_len) < 2:
            raise ValueError(f"Chronos patches ({seq_len} steps) are too short for dataset '{dataset_name}'.")

        for local_epoch in range(int(epochs_per_dataset)):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device=device,
                pred_len=pred_len,
                scaler=scaler,
                max_grad_norm=max_grad_norm,
                amp_enabled=amp_enabled,
            )

            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device=device,
                pred_len=pred_len,
                denorm_stats=denorm_stats,
                epoch=global_epoch,
                plot_dir=checkpoint_dir / "plots" / dataset_name,
                experiment=experiment,
            )

            val_loss = val_metrics[0] if val_metrics is not None else None

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
            elif scheduler is not None:
                scheduler.step()

            if experiment is not None:
                experiment.log_metric("train_loss", train_loss, step=global_epoch + 1)
                if val_loss is not None:
                    experiment.log_metric("val_loss", val_loss, step=global_epoch + 1)
                experiment.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=global_epoch + 1)
                experiment.log_metric("dataset_index", dataset_idx, step=global_epoch + 1)

            metric = val_loss if val_loss is not None else train_loss
            print(
                f"Global epoch {global_epoch + 1} | dataset={dataset_name}"
                f" | train={train_loss:.6e}"
                + (
                    f" | val_mse={val_metrics[0]:.6e} | val_mae={val_metrics[1]:.6e}"
                    if val_metrics is not None
                    else ""
                )
            )

            state = {
                "epoch": global_epoch + 1,
                "dataset": dataset_name,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            torch.save(state, checkpoint_dir / "last.pt")
            if metric < best_metric:
                best_metric = metric
                torch.save({**state, "best_metric": best_metric}, checkpoint_dir / "best.pt")

            global_epoch += 1


def train_temporal_encoder(
    *,
    config: tu.ExperimentConfig,
    training_cfg: dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    pred_len: int,
    feature_dim: int,
    checkpoint_dir: Path,
    experiment: Optional[comet_ml.Experiment] = None,
) -> None:
    encoder = tu.build_encoder_from_config(config.model)
    model = ChronosForecastModel(encoder, feature_dim, target_dim=feature_dim, pred_len=pred_len)
    _train_encoder(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        device=device,
        epochs=epochs,
        pred_len=pred_len,
        checkpoint_dir=checkpoint_dir,
        experiment=experiment,
    )


def train_visual_encoder(
    *,
    config: tu.ExperimentConfig,
    training_cfg: dict,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    pred_len: int,
    feature_dim: int,
    checkpoint_dir: Path,
    experiment: Optional[comet_ml.Experiment] = None,
) -> None:
    encoder = tu.build_visual_encoder_from_config(config.model)
    model = ChronosForecastModel(encoder, feature_dim, target_dim=feature_dim, pred_len=pred_len)
    _train_encoder(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=training_cfg,
        device=device,
        epochs=epochs,
        pred_len=pred_len,
        checkpoint_dir=checkpoint_dir,
        experiment=experiment,
    )


def main() -> None:
    # Initialize Comet ML experiment from config
    from comet_utils import create_comet_experiment
    experiment = create_comet_experiment("chronos_supervised")
    
    default_cfg = Path(__file__).resolve().parent / "configs" / "chronos_supervised.yaml"
    config_path = resolve_path(Path.cwd(), default_cfg)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {default_cfg}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")
    
    # Log configuration to Comet
    experiment.log_parameters({
        "seed": config.seed,
        "config_file": str(config_path),
    })

    data_cfg = config.data

    training_cfg = config.training
    pred_len = int(training_cfg.get("pred_len", 96))
    epochs = int(training_cfg.get("epochs", 100))

    checkpoint_root = resolve_checkpoint_dir(config, config_path, None)
    # Force temporal encoder training for Chronos supervised runs.
    encoder_choice = "temporal"
    checkpoint_dir = (checkpoint_root / encoder_choice).resolve()
    print(f"Checkpoints: {checkpoint_dir}")

    # Force Chronos normalization to global min-max (override YAML config).
    # This affects the Chronos patched dataset created by `load_cronos_time_series_dataset`.
    try:
        import yaml

        cronos_config = data_cfg.get("cronos_config")
        if cronos_config is None:
            cronos_config = config_path.parent / "cronos_loader_example.yaml"
        cronos_config = resolve_path(config_path.parent, cronos_config)
        if cronos_config is None or not Path(cronos_config).exists():
            raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

        override_path = _write_cronos_minmax_override(
            Path(cronos_config),
            out_path=checkpoint_dir / "configs" / "cronos_loader_minmax.yaml",
        )
        data_cfg = dict(data_cfg)
        data_cfg["cronos_config"] = str(override_path)
        if experiment is not None:
            experiment.log_parameter("cronos_config_override", str(override_path))
            experiment.log_parameter("normalize_mode", "global_minmax")
    except Exception as exc:
        print(f"Warning: could not create Chronos min-max override config: {exc}")

    # Log encoder type and dataset info to Comet
    experiment.log_parameters({
        "encoder_type": encoder_choice,
        "pred_len": pred_len,
    })
    
    # Read datasets_to_load from the Chronos loader config (data.cronos_config).
    dataset_names: list[str] = []
    try:
        import yaml

        cronos_config = data_cfg.get("cronos_config")
        if cronos_config is None:
            cronos_config = config_path.parent / "cronos_loader_example.yaml"
        cronos_config = resolve_path(config_path.parent, cronos_config)
        if cronos_config is None or not Path(cronos_config).exists():
            raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

        with open(cronos_config, "r", encoding="utf-8") as handle:
            raw_cfg = yaml.safe_load(handle) or {}

        def _find(cfg, key):
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

        names = _find(raw_cfg, "datasets_to_load")
        if isinstance(names, (list, tuple)):
            dataset_names = [str(x) for x in names]
    except Exception as exc:
        print(f"Warning: could not read datasets_to_load from Chronos loader config: {exc}")

    # Train sequentially over all datasets listed in the Chronos loader YAML.
    if dataset_names:
        epochs_per_dataset = int(training_cfg.get("epochs_per_dataset", 10))
        if epochs_per_dataset <= 0:
            raise ValueError("training.epochs_per_dataset must be a positive integer")

        # Build model once and train continuously across datasets.
        # Infer feature_dim from the first dataset.
        first_cfg = dict(data_cfg)
        first_cfg["dataset_name"] = dataset_names[0]
        # Do not override Chronos loader YAML patch_length (avoids accidental 1024).
        first_cfg["patch_length"] = None
        first_dataset = prepare_dataset(config_path, first_cfg)
        first_train, _ = split_dataset(first_dataset, val_ratio=float(first_cfg.get("val_ratio", 0.1)), seed=config.seed)
        first_loader, _ = build_dataloaders(
            first_train,
            None,
            batch_size=int(first_cfg.get("batch_size", 128)),
            val_batch_size=int(first_cfg.get("batch_size", 128)),
            num_workers=int(first_cfg.get("num_workers", 0)),
            pin_memory=bool(first_cfg.get("pin_memory", False)),
        )
        sample_batch = next(iter(first_loader))
        sample_seq = u.prepare_sequence(u.extract_sequence(sample_batch))
        feature_dim = sample_seq.shape[-1]
        experiment.log_parameter("feature_dim", feature_dim)

        encoder = tu.build_encoder_from_config(config.model)
        model = ChronosForecastModel(encoder, feature_dim, target_dim=feature_dim, pred_len=pred_len)

        _train_encoder_sequential_datasets(
            model,
            config_path=config_path,
            data_cfg=data_cfg,
            training_cfg=training_cfg,
            device=device,
            pred_len=pred_len,
            checkpoint_dir=checkpoint_dir,
            dataset_names=dataset_names,
            epochs_per_dataset=epochs_per_dataset,
            seed=config.seed,
            experiment=experiment,
        )
    else:
        raise ValueError(
            "No datasets_to_load found in the Chronos loader YAML. "
            "Set data.cronos_config in chronos_supervised.yaml or edit src/configs/cronos_loader_example.yaml."
        )
    
    # End Comet experiment
    experiment.end()


if __name__ == "__main__":
    main()
