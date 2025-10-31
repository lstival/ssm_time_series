"""Lightweight CLIP-style training loop using Chronos datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import training_utils as tu
import util as u
from moco_training import (
    build_dataloaders,
    infer_feature_dim,
    prepare_dataset,
    resolve_path,
    split_dataset,
    resolve_checkpoint_dir
)


def main(argv: Optional[Iterable[str]] = None) -> None:
    default_cfg = Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml"
    config_path = resolve_path(Path.cwd(), default_cfg)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {default_cfg}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    checkpoint_dir = resolve_checkpoint_dir(config, config_path, None)
    print(f"Checkpoints: {checkpoint_dir}")

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

    u.run_clip_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        projection_head=projection_head,
        visual_projection_head=visual_projection_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=100,
    )


if __name__ == "__main__":
    main()

