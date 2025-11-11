"""Lightweight CLIP-style training loop using Chronos datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import torch

import training_utils as tu
import util as u
from moco_training import (
    build_dataloaders,
    infer_feature_dim,
    prepare_dataset,
    resolve_path,
    resolve_checkpoint_dir,
    split_dataset,
)


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

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 100))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.01))

    data_cfg = config.data
    dataset = prepare_dataset(config_path, data_cfg)

    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=config.seed)

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    feature_dim = infer_feature_dim(train_loader)
    print(f"Inferred feature dimension: {feature_dim}")

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

    val_loader = val_loader if val_loader is not None and len(val_loader) > 0 else None

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
    )


if __name__ == "__main__":
    main()

