from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Tuple, Union

BatchType = Union[Tuple[torch.Tensor, ...], dict[str, torch.Tensor], torch.Tensor]

def extract_sequence(batch: BatchType) -> torch.Tensor:
    """Normalize different batch structures to a single tensor (batch, time, features)."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        return batch.get("x") or batch.get("data") or next(iter(batch.values()))
    return batch

def random_time_mask(x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """Apply random masking along the time dimension."""
    mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > drop_prob
    return x * mask

def random_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Add random Gaussian noise."""
    return x + torch.randn_like(x) * sigma

def random_scaling(x: torch.Tensor, low: float = 0.9, high: float = 1.1) -> torch.Tensor:
    """Apply random scaling."""
    scales = torch.empty(x.shape[0], 1, x.shape[2], device=x.device).uniform_(low, high)
    return x * scales

def create_contrastive_views(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create two augmented views of the same sequence for contrastive learning."""
    x1 = random_time_mask(random_jitter(x))
    x2 = random_scaling(random_jitter(x))
    return x1, x2

def mask_time_series(
    x: torch.Tensor,
    mask_prob: float = 0.75,
    mask_value: float = 0.0
) -> torch.Tensor:
    """Return a masked version of a batch of time-series."""
    B, T, F = x.shape
    mask = torch.rand((B, T, 1), device=x.device) > mask_prob
    return x * mask + (1 - mask.float()) * mask_value
