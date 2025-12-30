from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ssm_time_series.data.loader import TimeSeriesDataModule

BatchType = Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor], torch.Tensor]


def _parse_datasets(value: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Parse dataset names from various input formats."""
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return list(value)


def build_time_series_dataloaders(
    *,
    data_dir: str | Path,
    filename: Optional[str] = None,
    dataset_name: Optional[str] = None,
    batch_size: int = 128,
    val_batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    normalize: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    train: bool = True,
    val: bool = True,
    test: bool = False,
    dataset_type: str = "icml",
    datasets: Optional[Union[str, Sequence[str]]] = None,
    val_split: Optional[float] = None,
    seed: int = 42,
    cronos_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val dataloaders for ICML or Chronos datasets."""
    # Note: Chronos handling would go here or in a specialized builder
    
    module = TimeSeriesDataModule(
        data_dir=str(data_dir),
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio if val_split is None else val_split,
        train=train,
        val=val,
        test=test,
    )
    module.setup()
    return module.get_dataloaders()


def extract_sequence(batch: BatchType) -> torch.Tensor:
    """Normalize different batch structures to a single tensor."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, dict):
        return batch.get("x") or batch.get("data") or next(iter(batch.values()))
    return batch


def prepare_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Ensure the sequence tensor has shape (batch, time, features)."""
    if seq.ndim == 2:
        return seq.unsqueeze(-1)
    return seq


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
    """Create two augmented views of the same sequence."""
    x1 = random_time_mask(random_jitter(x))
    x2 = random_scaling(random_jitter(x))
    return x1, x2


def mask_time_series(
    x: torch.Tensor,
    mask_prob: float = 0.75,
    mask_value: float = 0.0,
    exact_fraction: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Return a masked version of a batch of time-series."""
    if seed is not None:
        torch.manual_seed(seed)
    B, T, F = x.shape
    if exact_fraction:
        num_mask = int(mask_prob * T)
        mask = torch.ones((B, T), device=x.device)
        for i in range(B):
            perm = torch.randperm(T, device=x.device)
            mask[i, perm[:num_mask]] = 0
        mask = mask.unsqueeze(-1)
    else:
        mask = torch.rand((B, T, 1), device=x.device) > mask_prob
    
    return x * mask + (1 - mask.float()) * mask_value


def simple_interpolation(time_series: Union[torch.Tensor, np.ndarray], target_size: int) -> Union[torch.Tensor, np.ndarray]:
    """Interpolate a 2D time series to target_size."""
    import numpy as np
    is_torch = isinstance(time_series, torch.Tensor)
    if is_torch:
        device = time_series.device
        dtype = time_series.dtype
        ts = time_series.detach().cpu().numpy()
    else:
        ts = np.asarray(time_series)
        
    T, F = ts.shape
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_size)
    
    new_ts = np.zeros((target_size, F), dtype=ts.dtype)
    for i in range(F):
        new_ts[:, i] = np.interp(x_new, x_old, ts[:, i])
        
    if is_torch:
        return torch.from_numpy(new_ts).to(device=device, dtype=dtype)
    return new_ts
