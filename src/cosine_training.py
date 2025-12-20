"""Lightweight CLIP-style training loop using Chronos datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
import yaml

import comet_ml
import torch
from torch.utils.data import DataLoader

import training_utils as tu
import util as u
from moco_training import (
    resolve_path,
    resolve_checkpoint_dir,
)


def _resolve_data_root(config_path: Path, candidate: Optional[Path | str]) -> Path:
    resolved = resolve_path(config_path.parent, candidate)
    if resolved is not None:
        return resolved
    default_root = config_path.parent.parent / "data"
    return default_root.resolve()


def _build_time_series_loaders(
    config_path: Path,
    data_cfg: Dict[str, object],
    *,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, object]]:
    dataset_type = str(data_cfg.get("dataset_type", "cronos")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    data_root = _resolve_data_root(config_path, data_cfg.get("data_dir"))
    normalize = bool(data_cfg.get("normalize", True))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio_cfg = data_cfg.get("val_ratio")
    val_ratio = float(val_ratio_cfg) if val_ratio_cfg is not None else 0.2
    if dataset_type == "cronos" and val_ratio <= 0.0:
        val_ratio = 0.2

    reverse_transform = bool(data_cfg.get("reverse_transform", False))
    cronos_kwargs: Dict[str, object] = dict(data_cfg.get("cronos_kwargs", {}) or {})

    datasets_spec = data_cfg.get("datasets")
    dataset_name = data_cfg.get("dataset_name")

    if dataset_type == "cronos":
        if not datasets_spec and dataset_name is None:
            config_candidates: Sequence[object] = (
                data_cfg.get("cronos_config"),
                config_path.parent / "cronos_loader_example.yaml",
            )
            cronos_config_path: Optional[Path] = None
            for candidate in config_candidates:
                resolved = resolve_path(config_path.parent, candidate) if candidate is not None else None
                if resolved is not None and resolved.exists():
                    cronos_config_path = resolved
                    break
            if cronos_config_path is not None and cronos_config_path.exists():
                with cronos_config_path.open("r", encoding="utf-8") as handle:
                    cronos_raw = yaml.safe_load(handle) or {}
                raw_datasets = cronos_raw.get("datasets_to_load")
                if isinstance(raw_datasets, Sequence) and not isinstance(raw_datasets, (str, bytes)):
                    datasets_spec = [str(item) for item in raw_datasets if str(item).strip()]
                repo_id = cronos_raw.get("repo_id")
                if repo_id is not None:
                    cronos_kwargs.setdefault("repo_id", repo_id)
                target_dtype = cronos_raw.get("target_dtype")
                if target_dtype is not None:
                    cronos_kwargs.setdefault("target_dtype", target_dtype)

                # Prefer global standard normalization unless explicitly configured otherwise.
                normalize_mode = cronos_raw.get("normalize_mode")
                if normalize_mode is None and bool(cronos_raw.get("normalize", True)):
                    normalize_mode = "global_standard"
                if normalize_mode is not None:
                    cronos_kwargs.setdefault("normalize_mode", str(normalize_mode))
                config_load_kwargs = cronos_raw.get("load_kwargs") or {}
                if isinstance(config_load_kwargs, dict):
                    load_kwargs = cronos_kwargs.setdefault("load_kwargs", {})
                    load_kwargs.update(config_load_kwargs)
            else:
                print(
                    "Warning: Cronos dataset list not provided and cronos_config file not found; training data may be empty."
                )

        load_kwargs = cronos_kwargs.setdefault("load_kwargs", {})
        load_kwargs.setdefault("offline_cache_dir", str(data_root))
        load_kwargs.setdefault("force_offline", True)

    train_loader, val_loader = u.build_time_series_dataloaders(
        data_dir=str(data_root),
        filename=data_cfg.get("filename"),
        dataset_name=dataset_name,
        datasets=datasets_spec,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        dataset_type=dataset_type,
        val_split=data_cfg.get("val_split"),
        seed=seed,
        cronos_kwargs=cronos_kwargs,
    )
    resolved = {
        "dataset_type": dataset_type,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "normalize": normalize,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "data_root": str(data_root),
        "reverse_transform": reverse_transform,
        "datasets": list(datasets_spec) if isinstance(datasets_spec, Sequence) and not isinstance(datasets_spec, (str, bytes)) else datasets_spec,
    }
    return train_loader, val_loader, resolved


def _infer_feature_dim(loader: DataLoader) -> Tuple[int, int]:
    iterator = iter(loader)
    try:
        sample = next(iterator)
    except StopIteration as exc:
        raise ValueError("Training data loader produced no batches.") from exc
    seq = u.prepare_sequence(u.extract_sequence(sample))
    return int(seq.shape[-1]), int(seq.shape[1])


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP-style training using Chronos datasets")
    default_cfg = Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory or file to resume from (expects *_last.pt files).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    
    # Initialize Comet ML experiment from config
    from comet_utils import create_comet_experiment
    experiment = create_comet_experiment("cosine_clip")

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")
    
    # Log configuration to Comet
    experiment.log_parameters({
        "seed": config.seed,
        "config_file": str(config_path),
        "device": str(device),
    })

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 100))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.01))
    
    # Log training hyperparameters to Comet
    experiment.log_parameters({
        "epochs": epochs,
        "noise_std": noise_std,
        "learning_rate": float(training_cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
    })

    data_cfg = config.data
    train_loader, val_loader, loader_meta = _build_time_series_loaders(
        config_path,
        data_cfg,
        seed=config.seed,
    )

    batch_size = int(loader_meta["batch_size"])
    num_workers = int(loader_meta["num_workers"])
    pin_memory = bool(loader_meta["pin_memory"])
    val_ratio = float(loader_meta["val_ratio"])
    dataset_type = str(loader_meta["dataset_type"])
    reverse_transform_flag = bool(loader_meta["reverse_transform"])

    if reverse_transform_flag:
        print(
            "Reverse transform flag enabled in config; contrastive training keeps normalized series during optimisation."
        )

    # Log data configuration to Comet
    experiment.log_parameters({
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "val_ratio": val_ratio,
        "dataset_type": dataset_type,
        "reverse_transform_requested": reverse_transform_flag,
    })

    feature_dim, sequence_length = _infer_feature_dim(train_loader)
    print(f"Inferred feature dimension: {feature_dim}")

    experiment.log_parameters({
        "feature_dim": feature_dim,
        "sequence_length": sequence_length,
    })

    encoder = tu.build_encoder_from_config(config.model)
    visual_encoder = tu.build_visual_encoder_from_config(config.model)
    projection_head = u.build_projection_head(encoder)
    visual_projection_head = u.build_projection_head(visual_encoder)

    params = (
        list(encoder.parameters())
        + list(visual_encoder.parameters())
        + list(projection_head.parameters())
        + list(visual_projection_head.parameters())
    )
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    resume_dir: Optional[Path] = None
    initial_epoch = 0
    best_loss: Optional[float] = None

    if args.resume_checkpoint is not None:
        resume_candidate = resolve_path(Path.cwd(), args.resume_checkpoint)
        if resume_candidate is None:
            raise FileNotFoundError(f"Unable to resolve resume path: {args.resume_checkpoint}")

        resume_dir = resume_candidate if resume_candidate.is_dir() else resume_candidate.parent
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

        def _load_component(name: str, module: torch.nn.Module) -> dict:
            path = resume_dir / f"{name}_last.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint file: {path}")
            state = torch.load(path, map_location="cpu")
            module.load_state_dict(state["model_state_dict"])
            return state

        time_series_state = _load_component("time_series", encoder)
        _load_component("visual_encoder", visual_encoder)
        _load_component("time_series_projection", projection_head)
        _load_component("visual_projection", visual_projection_head)

        optimizer.load_state_dict(time_series_state["optimizer_state_dict"])
        stored_epoch = int(time_series_state.get("epoch", 0))
        initial_epoch = min(epochs, stored_epoch + 1)

        best_path = resume_dir / "time_series_best.pt"
        best_candidate = None
        if best_path.exists():
            best_state = torch.load(best_path, map_location="cpu")
            best_candidate = best_state.get("loss")
        if best_candidate is None:
            best_candidate = time_series_state.get("loss")
        if best_candidate is not None:
            best_loss = float(best_candidate)

        resume_display_epoch = min(epochs, stored_epoch + 1)
        print(f"Resuming from checkpoint: {resume_dir.resolve()} (epoch {resume_display_epoch}).")

    checkpoint_dir = (
        resume_dir.resolve()
        if resume_dir is not None
        else resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    )
    print(f"Checkpoints: {checkpoint_dir}")

    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    u.run_clip_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        projection_head=projection_head,
        visual_projection_head=visual_projection_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        optimizer=optimizer,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
        experiment=experiment,
    )
    
    # End Comet experiment
    experiment.end()


if __name__ == "__main__":
    main()

