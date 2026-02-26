from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


class MoCoProjectionHead(nn.Module):
    """Projection head for MoCo model."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def load_encoder_checkpoint(
    encoder: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Load weights from a checkpoint file into an encoder."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove prefixes if necessary (e.g., from DataParallel or wrappers)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_state_dict[k[8:]] = v
        elif k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    encoder.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded encoder checkpoint from {checkpoint_path}")


def default_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_run_directory(base_dir: str | Path, prefix: str) -> Path:
    """Create and return a timestamped run directory."""
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_and_save(
    optimizer: torch.optim.Optimizer,
    *,
    models: Dict[str, nn.Module],
    epoch: int,
    epochs: int,
    train_loss: float,
    val_loss: Optional[float],
    checkpoint_dir: Path,
    best_loss: float,
) -> float:
    """Log metrics and persist checkpoints for the provided models."""
    metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss if val_loss is not None else float("nan"),
    }

    # Save checkpoint
    checkpoint = {
        "epoch": epoch + 1,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    for name, model in models.items():
        checkpoint[f"{name}_state_dict"] = model.state_dict()

    last_path = checkpoint_dir / "last_checkpoint.pt"
    torch.save(checkpoint, last_path)

    current_loss = val_loss if val_loss is not None else train_loss
    if current_loss < best_loss:
        best_loss = current_loss
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"  → Saved new best model (loss: {best_loss:.4f})")

    return best_loss


def build_projection_head(encoder: nn.Module) -> nn.Module:
    """Infer encoder output dimension and create a projection head."""
    # This is a heuristic based on the project's encoders
    if hasattr(encoder, "embedding_dim"):
        input_dim = encoder.embedding_dim
    elif hasattr(encoder, "model_dim"):
        input_dim = encoder.model_dim
    else:
        # Fallback to a common default
        input_dim = 128
        
    return MoCoProjectionHead(input_dim=input_dim, hidden_dim=input_dim * 2, output_dim=input_dim)


class SimpleEncoder(nn.Module):
    """A tiny encoder that averages over time and projects to an embedding."""

    def __init__(self, in_features: int, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        return self.proj(x.mean(dim=1))
