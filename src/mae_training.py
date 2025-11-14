"""Lightweight masked autoencoder training loop using Chronos datasets."""

from __future__ import annotations

import argparse
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
    resolve_checkpoint_dir,
    split_dataset,
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE training using Chronos datasets")
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
    parser.add_argument("--time-loss-weight", type=float, default=None)
    parser.add_argument("--visual-loss-weight", type=float, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


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
    val_loader: Optional[object] = None,
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int = 2,
    noise_std: float = 0.01,
    time_loss_weight: float = 0.5,
    visual_loss_weight: float = 0.5,
    optimizer: Optional[torch.optim.Optimizer] = None,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
) -> None:
    """Masked autoencoder training loop with weighted reconstruction losses."""

    if time_loss_weight < 0 or visual_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    weight_total = time_loss_weight + visual_loss_weight
    if weight_total == 0:
        raise ValueError("At least one loss weight must be positive")

    time_w = time_loss_weight / weight_total
    visual_w = visual_loss_weight / weight_total

    encoder.to(device)
    visual_encoder.to(device)
    decoder_head.to(device)
    visual_decoder_head.to(device)

    params = (
        list(encoder.parameters())
        + list(visual_encoder.parameters())
        + list(decoder_head.parameters())
        + list(visual_decoder_head.parameters())
    )
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=1e-3)

    # Move optimizer state to device
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)

    best_metric = float("inf") if best_loss is None else float(best_loss)
    start_epoch = max(0, int(initial_epoch))
    if start_epoch >= epochs:
        print(f"Requested epochs already completed ({start_epoch}/{epochs}).")
        return

    for epoch in range(start_epoch, epochs):
        encoder.train()
        visual_encoder.train()
        decoder_head.train()
        visual_decoder_head.train()
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

        train_loss = epoch_loss / batches if batches > 0 else float("nan")
        train_time_loss = epoch_time_loss / batches if batches > 0 else float("nan")
        train_visual_loss = epoch_visual_loss / batches if batches > 0 else float("nan")

        # Validation loop
        val_loss = None
        if val_loader is not None:
            encoder.eval()
            visual_encoder.eval()
            decoder_head.eval()
            visual_decoder_head.eval()

            val_epoch_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_seq = u.prepare_sequence(u.extract_sequence(val_batch)).to(device)

                    val_series_view = val_seq.swapaxes(1, 2)
                    val_positive_view = u.make_positive_view(
                        val_series_view + noise_std * torch.randn_like(val_series_view),
                        noise_std=noise_std,
                    )

                    val_time_target = val_series_view
                    val_visual_target = val_positive_view

                    val_masked_time = u.mask_time_series(val_series_view)
                    val_masked_visual = u.mask_time_series(val_positive_view)

                    val_time_encoded = encoder(val_masked_time)
                    val_visual_encoded = visual_encoder(val_masked_visual)

                    val_time_recon = decoder_head(val_time_encoded)
                    val_visual_recon = visual_decoder_head(val_visual_encoded)

                    val_time_loss = nn.functional.mse_loss(val_time_recon, val_time_target)
                    val_visual_loss = nn.functional.mse_loss(val_visual_recon, val_visual_target)
                    val_batch_loss = time_w * val_time_loss + visual_w * val_visual_loss

                    val_epoch_loss += float(val_batch_loss.item())
                    val_batches += 1

            if val_batches > 0:
                val_loss = val_epoch_loss / val_batches

            encoder.train()
            visual_encoder.train()
            decoder_head.train()
            visual_decoder_head.train()

        # Save checkpoints using the same system as cosine_training.py
        models_to_save = {
            "time_series": encoder,
            "visual_encoder": visual_encoder,
            "time_series_decoder": decoder_head,
            "visual_decoder": visual_decoder_head,
        }

        best_metric = u.log_and_save(
            optimizer,
            models=models_to_save,
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint_dir=checkpoint_dir,
            best_loss=best_metric,
        )


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
    args = parse_args(argv)

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 2))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.01))
    time_loss_weight = args.time_loss_weight if args.time_loss_weight is not None else float(training_cfg.get("time_loss_weight", 0.5))
    visual_loss_weight = args.visual_loss_weight if args.visual_loss_weight is not None else float(training_cfg.get("visual_loss_weight", 0.5))

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

    params = (
        list(encoder.parameters())
        + list(visual_encoder.parameters())
        + list(decoder_head.parameters())
        + list(visual_decoder_head.parameters())
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
        _load_component("time_series_decoder", decoder_head)
        _load_component("visual_decoder", visual_decoder_head)

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

    run_mae_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        decoder_head=decoder_head,
        visual_decoder_head=visual_decoder_head,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        time_loss_weight=time_loss_weight,
        visual_loss_weight=visual_loss_weight,
        optimizer=optimizer,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
    )


if __name__ == "__main__":
    main()

