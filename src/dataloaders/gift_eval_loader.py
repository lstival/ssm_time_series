"""Dataloader for GIFT-Eval benchmark.

Usage:
    from dataloaders.gift_eval_loader import build_gift_eval_dataloader
    
    loader = build_gift_eval_dataloader("ett1_H_short", context_length=336, prediction_length=96)
    batch = next(iter(loader))
    # batch['target'] -> context window (B, L, 1)
    # batch['future'] -> target window (B, H, 1)
"""

from __future__ import annotations
import os
import torch
import datasets
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Sequence, Optional, Dict, Tuple, Union

logger = logging.getLogger(__name__)

GIFT_EVAL_REPO_ID = "Salesforce/GiftEvalParquet"
DEFAULT_CACHE_DIR = "/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"

class GiftEvalDataset(Dataset):
    """
    Dataset for GIFT-Eval subsets (from GiftEvalParquet).
    Expects features: 'history_value' (context) and 'future_value' (prediction).
    """
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        context_length: int = 336,
        prediction_length: int = 96,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.dataset = hf_dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dtype = torch_dtype

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[index]

        # history_value and future_value are lists of floats
        history = np.asarray(item["history_value"], dtype=np.float32)
        future = np.asarray(item["future_value"], dtype=np.float32)

        # Replace NaN with 0 (standard for intermittent/sparse series like car_parts)
        history = np.nan_to_num(history, nan=0.0)
        future = np.nan_to_num(future, nan=0.0)

        # Ensure context length (L)
        if len(history) > self.context_length:
            history = history[-self.context_length:]
        elif len(history) < self.context_length:
            # Left-pad with zeros if too short
            pad = np.zeros(self.context_length, dtype=np.float32)
            pad[-len(history):] = history
            history = pad

        # Ensure prediction length (H)
        if len(future) > self.prediction_length:
            future = future[:self.prediction_length]

        # Per-series z-score normalisation using context statistics.
        # Keeps MSE in normalised space (comparable across subsets and with SEMPO/Chronos).
        # Avoids exploding MSE on high-amplitude subsets (bizitobs, m_dense).
        mu = history.mean()
        sigma = history.std() + 1e-8
        history = (history - mu) / sigma
        future  = (future  - mu) / sigma

        x = torch.as_tensor(history, dtype=self.dtype).unsqueeze(-1) # (L, 1)
        y = torch.as_tensor(future, dtype=self.dtype).unsqueeze(-1)  # (H, 1)

        return {
            "target": x,
            "future": y,
            "item_id": item.get("item_id", "unknown")
        }

def load_gift_eval_hf(
    subset_name: str,
    split: str = "train",
    repo_id: str = GIFT_EVAL_REPO_ID,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    force_offline: bool = False,
) -> datasets.Dataset:
    """Load a specific subset of GIFT-Eval from HuggingFace."""
    
    # Save original env to restore later
    prev_cache = os.environ.get("HF_DATASETS_CACHE")
    prev_offline = os.environ.get("HF_DATASETS_OFFLINE")
    prev_hub = os.environ.get("HF_HUB_OFFLINE")

    if cache_dir:
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        
    if force_offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info(f"Offline mode — loading {subset_name} from: {cache_dir}")

    try:
        ds = datasets.load_dataset(repo_id, subset_name, split=split)
        logger.info(f"Loaded GIFT-Eval subset '{subset_name}': {len(ds):,} rows")
        return ds
    finally:
        # Restore env
        if prev_cache is not None:
            os.environ["HF_DATASETS_CACHE"] = prev_cache
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)
            
        if prev_offline is not None:
            os.environ["HF_DATASETS_OFFLINE"] = prev_offline
        else:
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            
        if prev_hub is not None:
            os.environ["HF_HUB_OFFLINE"] = prev_hub
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)

ALL_GIFT_SSL_SUBSETS = [
    "m_dense_H_long",
    "loop_seattle_H_long",
    "sz_taxi_H_short",
    "solar_H_long",
    "bizitobs_application_10S_long",
    "bizitobs_l2c_H_long",
    "bizitobs_service_10S_long",
    "car_parts_M_short",
    "jena_weather_H_long",
]


def load_gift_for_ssl(
    subset_names: Optional[Sequence[str]] = None,
    split: str = "train",
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    force_offline: bool = True,
    normalize_per_series: bool = True,
) -> datasets.Dataset:
    """Load GIFT-Eval train splits and expose them as a ``target``-column HF Dataset
    suitable for SSL pre-training (same schema as LOTSA/Chronos corpus).

    ``history_value`` (context window) → ``target`` list of floats.
    The ``future_value`` column is dropped; only context is used for SSL.
    """
    if subset_names is None:
        subset_names = ALL_GIFT_SSL_SUBSETS

    parts = []
    for name in subset_names:
        try:
            raw = load_gift_eval_hf(name, split=split, cache_dir=cache_dir,
                                    force_offline=force_offline)
            # Remap history_value → target; drop future_value
            def _remap(batch):
                targets = []
                for hist in batch["history_value"]:
                    arr = np.asarray(hist, dtype=np.float64)
                    arr = np.nan_to_num(arr, nan=0.0)
                    if normalize_per_series:
                        sigma = arr.std() + 1e-8
                        arr = (arr - arr.mean()) / sigma
                    # Wrap as (T, 1) list-of-lists to match LOTSA/local schema
                    targets.append([[float(v)] for v in arr])
                return {"target": targets}

            remapped = raw.map(_remap, batched=True, remove_columns=raw.column_names)
            parts.append(remapped)
            logger.info("GIFT SSL subset '%s': %d rows", name, len(remapped))
            print(f"Loaded GIFT-SSL ({name}): {len(remapped):,} rows")
        except Exception as exc:
            logger.warning("Could not load GIFT subset '%s': %s", name, exc)

    if not parts:
        raise RuntimeError("No GIFT subsets could be loaded for SSL pre-training.")
    return datasets.concatenate_datasets(parts)


def build_gift_eval_dataloader(
    subset_name: str,
    context_length: int = 336,
    prediction_length: int = 96,
    batch_size: int = 64,
    num_workers: int = 4,
    repo_id: str = GIFT_EVAL_REPO_ID,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    force_offline: bool = False,
) -> DataLoader:
    """Build a DataLoader for a GIFT-Eval subset."""
    ds_hf = load_gift_eval_hf(
        subset_name, 
        repo_id=repo_id, 
        cache_dir=cache_dir, 
        force_offline=force_offline
    )
    
    ds_torch = GiftEvalDataset(
        ds_hf, 
        context_length=context_length, 
        prediction_length=prediction_length
    )
    
    return DataLoader(
        ds_torch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO)
    print("Running GIFT-Eval Dataloader Smoke Test...")
    
    # Using ett1_H_short as it is relatively small
    try:
        loader = build_gift_eval_dataloader(
            "ett1_H_short", 
            context_length=96, 
            prediction_length=24,
            batch_size=8,
            num_workers=0,
            force_offline=False
        )
        
        batch = next(iter(loader))
        print("Batch keys:", batch.keys())
        print("Target shape (context):", batch["target"].shape) # (8, 96, 1)
        print("Future shape (target) :", batch["future"].shape) # (8, 24, 1)
        print("First item ID:", batch["item_id"][0])
        print("Smoke test PASSED!")
    except Exception as e:
        print(f"Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
