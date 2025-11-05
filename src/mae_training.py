"""Lightweight masked autoencoder training loop using Chronos datasets."""

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


class ReconstructionDecoder(nn.Module):
    """Simple MLP decoder that reconstructs masked sequences from embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        channels: int,
        sequence_length: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        hidden_size = int(hidden_dim or max(embedding_dim * 2, channels * 8))
        self.channels = channels
        self.sequence_length = sequence_length
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, channels * sequence_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(x)
        return out.view(x.size(0), self.channels, self.sequence_length)

def run_mae_training(
    *,
    encoder: nn.Module,
    visual_encoder: nn.Module,
    decoder_head: nn.Module,
    visual_decoder_head: nn.Module,
    train_loader,
    device: torch.device,
    epochs: int = 2,
    noise_std: float = 0.01,
    time_loss_weight: float = 0.5,
    visual_loss_weight: float = 0.5,
) -> None:
    """Masked autoencoder training loop with weighted reconstruction losses."""

    if time_loss_weight < 0 or visual_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    weight_total = time_loss_weight + visual_loss_weight
    if weight_total == 0:
        raise ValueError("At least one loss weight must be positive")

    time_w = time_loss_weight / weight_total
    visual_w = visual_loss_weight / weight_total

    encoder.to(device).train()
    visual_encoder.to(device).train()
    decoder_head.to(device).train()
    visual_decoder_head.to(device).train()

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(visual_encoder.parameters())
        + list(decoder_head.parameters())
        + list(visual_decoder_head.parameters()),
        lr=1e-3,
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_time_loss = 0.0
        epoch_visual_loss = 0.0
        batches = 0

        total = len(train_loader) if hasattr(train_loader, "__len__") else None
        desc = f"Epoch {epoch + 1}/{epochs}"
        with tqdm(train_loader, desc=desc, total=total) as pbar:
            for batch in pbar:
                seq = u.prepare_sequence(u.extract_sequence(batch)).to(device)

                series_view = seq.swapaxes(1, 2)
                positive_view = u.make_positive_view(
                    series_view + noise_std * torch.randn_like(series_view),
                    noise_std=noise_std,
                )

                time_target = series_view
                visual_target = positive_view

                optimizer.zero_grad()

                masked_time = u.mask_time_series(series_view)
                masked_visual = u.mask_time_series(positive_view)

                time_encoded = encoder(masked_time)
                visual_encoded = visual_encoder(masked_visual)

                time_recon = decoder_head(time_encoded)
                visual_recon = visual_decoder_head(visual_encoded)

                time_loss = nn.functional.mse_loss(time_recon, time_target)
                visual_loss = nn.functional.mse_loss(visual_recon, visual_target)
                loss = time_w * time_loss + visual_w * visual_loss

                loss.backward()
                optimizer.step()

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                epoch_time_loss += float(time_loss.item())
                epoch_visual_loss += float(visual_loss.item())
                batches += 1

                pbar.set_postfix(
                    total_loss=f"{(epoch_loss / batches):.4f}",
                    time_loss=f"{(epoch_time_loss / batches):.4f}",
                    visual_loss=f"{(epoch_visual_loss / batches):.4f}",
                )

        if batches > 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - Total: {epoch_loss / batches:.4f} | "
                f"Time: {epoch_time_loss / batches:.4f} | Visual: {epoch_visual_loss / batches:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{epochs} - no batches processed")


def build_decoder_head(
    encoder: nn.Module,
    *,
    channels: int,
    sequence_length: int,
    hidden_dim: Optional[int] = None,
) -> nn.Module:
    """Infer encoder embedding size and create a reconstruction decoder."""

    try:
        embedding_dim = encoder.output_proj.out_features
    except AttributeError as exc:
        raise RuntimeError("Encoder is expected to expose output_proj layer with out_features") from exc

    return ReconstructionDecoder(
        embedding_dim=embedding_dim,
        channels=channels,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
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

    try:
        sample_batch = next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError("Training loader is empty; cannot infer reconstruction shape") from exc

    sample_seq = u.prepare_sequence(u.extract_sequence(sample_batch))
    sample_view = sample_seq.swapaxes(1, 2)
    channels = int(sample_view.shape[1])
    sequence_length = int(sample_view.shape[2])
    del sample_batch, sample_seq, sample_view
    print(f"Reconstruction target shape: channels={channels}, sequence_length={sequence_length}")

    decoder_head = build_decoder_head(
        encoder,
        channels=channels,
        sequence_length=sequence_length,
    )
    visual_decoder_head = build_decoder_head(
        visual_encoder,
        channels=channels,
        sequence_length=sequence_length,
    )

    training_cfg = config.training
    epochs = int(training_cfg.get("epochs", 2))
    time_loss_weight = float(training_cfg.get("time_loss_weight", 0.5))
    visual_loss_weight = float(training_cfg.get("visual_loss_weight", 0.5))
    noise_std = float(training_cfg.get("noise_std", 0.01))

    run_mae_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        decoder_head=decoder_head,
        visual_decoder_head=visual_decoder_head,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        noise_std=noise_std,
        time_loss_weight=time_loss_weight,
        visual_loss_weight=visual_loss_weight,
    )


if __name__ == "__main__":
    main()

