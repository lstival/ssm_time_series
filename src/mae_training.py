"""Lightweight cosine-similarity training loop using Chronos datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

import training_utils as tu
import util as u
from tqdm.auto import tqdm
from moco_training import (
    build_dataloaders,
    infer_feature_dim,
    prepare_dataset,
    resolve_path,
    split_dataset,
)

def run_cosine_training(
    *,
    encoder: nn.Module,
    visual_encoder: nn.Module,
    projection_head: nn.Module,
    visual_projection_head: nn.Module,
    train_loader,
    device: torch.device,
    epochs: int = 2,
    noise_std: float = 0.01,
) -> None:
    """Training loop that maximizes cosine similarity between two noisy views.

    Optimizer: AdamW with lr=1e-3.
    Loss: 1 - mean(cosine_similarity(q_proj, k_proj))
    """

    encoder.to(device).train()
    visual_encoder.to(device).train()
    projection_head.to(device).train()
    visual_projection_head.to(device).train()

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(visual_encoder.parameters()) +
        list(projection_head.parameters()) + list(visual_projection_head.parameters()),
        lr=1e-3
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0

        total = len(train_loader) if hasattr(train_loader, "__len__") else None
        desc = f"Epoch {epoch + 1}/{epochs}"
        with tqdm(train_loader, desc=desc, total=total) as pbar:
            for batch in pbar:
                seq = u.prepare_sequence(u.extract_sequence(batch)).to(device)

                x_q = seq.swapaxes(1, 2)
                noise = noise_std * torch.randn_like(x_q)
                x_k = x_q + noise
                x_k = u.make_positive_view(x_k)

                optimizer.zero_grad()

                x_q = u.mask_time_series(x_q)
                x_k = u.mask_time_series(x_k)

                q_encoded = encoder(x_q)
                k_encoded = visual_encoder(x_k)

                q_proj = projection_head(q_encoded)
                k_proj = visual_projection_head(k_encoded)

                q_proj = nn.functional.normalize(q_proj, dim=1)
                k_proj = nn.functional.normalize(k_proj, dim=1)

                cosine_sim = nn.functional.cosine_similarity(q_proj, k_proj, dim=1)
                loss = (1.0 - cosine_sim).mean()
                # loss = clip_contrastive_loss(q_proj, k_proj)

                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                batches += 1

                pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{(epoch_loss / batches):.4f}")

        avg_loss = epoch_loss / batches if batches > 0 else float("nan")
        print(f"Epoch {epoch + 1}/{epochs} - Average loss: {avg_loss:.4f}")


def build_projection_head(encoder: nn.Module) -> nn.Module:
    """Infer encoder output dimension and create a projection head."""
    try:
        output_dim = encoder.final_norm._parameters["weight"].shape[0]
    except AttributeError as exc:
        raise RuntimeError("Encoder is expected to expose final_norm with learnable weight") from exc

    return u.MoCoProjectionHead(output_dim, output_dim, 128)


def main(argv: Optional[Iterable[str]] = None) -> None:
    default_cfg = Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml"
    config_path = resolve_path(Path.cwd(), default_cfg)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {default_cfg}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    data_cfg = config.data
    dataset = prepare_dataset(config_path, data_cfg)

    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    train_dataset, _ = split_dataset(dataset, val_ratio=val_ratio, seed=config.seed)

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader, _ = build_dataloaders(
        train_dataset,
        None,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    feature_dim = infer_feature_dim(train_loader)
    print(f"Inferred feature dimension: {feature_dim}")

    encoder = tu.build_encoder_from_config(config.model)
    visual_encoder = tu.build_visual_encoder_from_config(config.model)

    projection_head = build_projection_head(encoder)
    visual_projection_head = build_projection_head(visual_encoder)

    run_cosine_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        projection_head=projection_head,
        visual_projection_head=visual_projection_head,
        train_loader=train_loader,
        device=device,
        epochs=2,
    )


if __name__ == "__main__":
    main()

