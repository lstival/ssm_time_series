"""PyTorch DataLoaders sourced from Salesforce/lotsa_data for MambaVisualEncoder training.

The key difference from the Chronos loader is that this loader samples **fixed-length
windows** from each time series.  MambaVisualEncoder (cm_mamba) tokenises its input
internally, so feeding it fixed-length windows avoids padding artefacts and ensures
every recurrence-plot patch has the same resolution.

Batch format
------------
Each batch is a dict::

    {
        "target":  FloatTensor of shape (B, T, 1),   # ready for MambaVisualEncoder
        "lengths": LongTensor  of shape (B,),         # always == context_length
    }

For contrastive (CLIP-style) training pass ``two_views=True``.  The batch then also
contains a second independent window sampled from the same series::

    {
        "target":   FloatTensor (B, T, 1),
        "target2":  FloatTensor (B, T, 1),
        "lengths":  LongTensor  (B,),
    }

Usage
-----
    from dataloaders.lotsa_loader import build_lotsa_dataloaders

    train_dl, val_dl = build_lotsa_dataloaders(
        dataset_names=["m4_hourly", "ett_h1"],
        context_length=96,
        batch_size=256,
    )
    for batch in train_dl:
        x = batch["target"]          # (B, 96, 1)
        emb = visual_encoder(x)      # (B, embedding_dim)
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloaders.lotsa_dataset import (
    LOTSA_CACHE_DIR,
    LOTSA_DEFAULT_SUBSETS,
    LOTSA_REPO_ID,
    load_lotsa_datasets,
)

logger = logging.getLogger(__name__)

SplitSpec = Union[int, float]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LotsaWindowDataset(Dataset):
    """Sample fixed-length windows from Salesforce/lotsa_data series.

    Each ``__getitem__`` call returns one window drawn from a randomly
    selected position in the series.  Series shorter than ``context_length``
    are left-padded with zeros.

    Parameters
    ----------
    hf_dataset:
        A HuggingFace Dataset with at least a ``target`` column (1-D float array
        per row, as returned by ``load_lotsa_datasets``).
    context_length:
        Number of time steps per window.  Must match the value expected by
        your MambaVisualEncoder (typically 96, 192, or 336).
    two_views:
        If True, ``__getitem__`` returns a pair of independent windows
        sampled from the **same** series — useful for contrastive training.
    torch_dtype:
        Output tensor dtype (default float32).
    seed:
        Optional RNG seed for reproducibility (per-worker seeds are derived
        automatically from this value by the DataLoader).
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        *,
        context_length: int = 96,
        two_views: bool = False,
        torch_dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        self.dataset = hf_dataset
        self.context_length = context_length
        self.two_views = two_views
        self.dtype = torch_dtype
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_array(self, index: int) -> np.ndarray:
        sample = self.dataset[index]
        arr = np.asarray(sample["target"], dtype=np.float32).reshape(-1)
        return arr

    def _sample_window(self, arr: np.ndarray) -> np.ndarray:
        """Return a single window of length ``context_length`` from ``arr``."""
        T = len(arr)
        L = self.context_length
        if T <= L:
            # Left-pad with zeros so the series occupies the last L positions.
            pad = np.zeros(L, dtype=np.float32)
            pad[L - T:] = arr
            return pad
        # Random start position.
        start = int(self._rng.integers(0, T - L + 1))
        return arr[start: start + L].copy()

    def _to_tensor(self, window: np.ndarray) -> torch.Tensor:
        # Replace NaNs / Infs with 0.
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.as_tensor(window, dtype=self.dtype)
        return t.unsqueeze(-1)  # (T, 1) — MambaVisualEncoder expects (B, T, F)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        arr = self._get_array(index)
        w1 = self._to_tensor(self._sample_window(arr))
        if self.two_views:
            w2 = self._to_tensor(self._sample_window(arr))
            return {"target": w1, "target2": w2}
        return {"target": w1}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _lotsa_collate_fn(
    batch: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    target = torch.stack([b["target"] for b in batch])          # (B, T, 1)
    B, T, _ = target.shape
    result: Dict[str, torch.Tensor] = {
        "target": target,
        "lengths": torch.full((B,), T, dtype=torch.long),
    }
    if "target2" in batch[0]:
        result["target2"] = torch.stack([b["target2"] for b in batch])
    return result


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_lotsa_dataloaders(
    dataset_names: Optional[Sequence[str]] = None,
    *,
    context_length: int = 96,
    val_split: SplitSpec = 0.1,
    batch_size: int = 256,
    val_batch_size: Optional[int] = None,
    two_views: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    cache_dir: str = LOTSA_CACHE_DIR,
    repo_id: str = LOTSA_REPO_ID,
    normalize_per_series: bool = True,
    force_offline: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train / validation DataLoaders from Salesforce/lotsa_data.

    Parameters
    ----------
    dataset_names:
        Subset names to load.  Defaults to ``LOTSA_DEFAULT_SUBSETS`` if None.
    context_length:
        Fixed window length returned by each sample.  Should match the
        context length your MambaVisualEncoder is trained with.
    val_split:
        Fraction (float) or absolute count (int) of samples to hold out for
        validation.
    batch_size:
        Training batch size.
    val_batch_size:
        Validation batch size (defaults to ``batch_size``).
    two_views:
        If True, each sample returns two independent windows (contrastive mode).
    num_workers:
        DataLoader worker processes.
    pin_memory:
        Pin host memory for faster GPU transfer.
    drop_last:
        Drop the last incomplete training batch.
    cache_dir:
        Lustre no-backup HuggingFace cache directory.
    repo_id:
        HuggingFace repository identifier.
    normalize_per_series:
        Apply per-series min-max normalisation when loading.
    force_offline:
        Load only from the local cache (no network requests).
    seed:
        Random seed for train/val split and window sampling.

    Returns
    -------
    train_loader, val_loader : DataLoader
        Each batch is a dict with keys ``"target"`` *(B, T, 1)* and
        ``"lengths"`` *(B,)*.  When ``two_views=True``, ``"target2"`` *(B, T, 1)*
        is also present.
    """
    if dataset_names is None:
        dataset_names = LOTSA_DEFAULT_SUBSETS

    merged = load_lotsa_datasets(
        dataset_names,
        repo_id=repo_id,
        cache_dir=cache_dir,
        force_offline=force_offline,
        normalize_per_series=normalize_per_series,
    )

    total = len(merged)
    logger.info("Total series loaded: %d", total)

    split = merged.train_test_split(
        test_size=val_split if isinstance(val_split, float) else val_split / total,
        shuffle=True,
        seed=seed,
    )
    train_hf = split["train"]
    val_hf = split["test"]

    train_ds = LotsaWindowDataset(
        train_hf,
        context_length=context_length,
        two_views=two_views,
        seed=seed,
    )
    val_ds = LotsaWindowDataset(
        val_hf,
        context_length=context_length,
        two_views=False,   # single view for validation
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_lotsa_collate_fn,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size or batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_lotsa_collate_fn,
        worker_init_fn=_worker_init_fn,
    )

    logger.info(
        "Dataloaders ready — train: %d series, val: %d series, context_length: %d",
        len(train_ds),
        len(val_ds),
        context_length,
    )
    return train_loader, val_loader


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own RNG seed."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = info.seed % (2 ** 32)
    dataset = info.dataset
    if isinstance(dataset, LotsaWindowDataset):
        dataset._rng = np.random.default_rng(seed)


__all__ = [
    "LotsaWindowDataset",
    "build_lotsa_dataloaders",
]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    train_dl, val_dl = build_lotsa_dataloaders(
        dataset_names=["m4_daily"],
        context_length=96,
        batch_size=8,
        num_workers=0,
        force_offline=False,
        seed=0,
    )

    batch = next(iter(train_dl))
    print("target shape :", batch["target"].shape)    # (8, 96, 1)
    print("lengths      :", batch["lengths"])         # tensor([96, 96, ...])

    # Verify compatibility with MambaVisualEncoder
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from models.mamba_visual_encoder import MambaVisualEncoder

    encoder = MambaVisualEncoder(input_dim=32, model_dim=128, depth=2, embedding_dim=64)
    encoder.eval()
    with torch.no_grad():
        emb = encoder(batch["target"])
    print("embedding shape:", emb.shape)              # (8, 64)
