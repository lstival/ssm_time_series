"""Dataloader for GIFT-Eval benchmark.

Usage:
    from dataloaders.gift_eval_loader import build_gift_eval_dataloader

    # prediction_length=None (default) uses the native horizon from future_value,
    # consistent with the official GIFT-Eval evaluation protocol.
    loader = build_gift_eval_dataloader("m_dense_H_long", context_length=336)
    batch = next(iter(loader))
    # batch['target'] -> context window (B, L, 1)
    # batch['future'] -> target window (B, H, 1)  — H = native benchmark horizon
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

    prediction_length=None (default): use the native benchmark horizon from
    len(future_value), consistent with the official GIFT-Eval protocol.
    Pass an explicit int only when you need a fixed horizon (e.g. ablations).
    """
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        context_length: int = 336,
        prediction_length: Optional[int] = None,
        torch_dtype: torch.dtype = torch.float32,
        fd_mode: bool = False,
        stride: Optional[int] = None,
        sliding_window: bool = False,
    ):
        self.dataset = hf_dataset
        self.context_length = context_length
        self.prediction_length = prediction_length  # None → use native horizon per row
        self.dtype = torch_dtype
        self.fd_mode = fd_mode

        if sliding_window and stride is None:
            stride = 1
        self.stride = stride

        # Resolve a single representative prediction_length for sliding-window
        # pre-computation (native mode: read from first row).
        if self.stride is not None and self.stride > 0:
            pred_len = prediction_length if prediction_length is not None else len(self.dataset[0]["future_value"])
        else:
            pred_len = prediction_length  # not used in this branch

        self.window_info = []
        if self.stride is not None and self.stride > 0:
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                h_len = len(item["history_value"])
                f_len = len(item["future_value"])
                total_len = h_len + f_len
                win_total = context_length + pred_len

                if total_len >= win_total:
                    num_windows = (total_len - win_total) // stride + 1
                    for w in range(num_windows):
                        self.window_info.append((i, w * stride))
        else:
            for i in range(len(self.dataset)):
                self.window_info.append((i, -1))

    def __len__(self) -> int:
        return len(self.window_info)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        series_idx, offset = self.window_info[index]
        item = self.dataset[series_idx]

        history_full = np.asarray(item["history_value"], dtype=np.float32)
        future_full  = np.asarray(item["future_value"],  dtype=np.float32)

        # Native prediction length: use benchmark's future_value length unless overridden.
        pred_len = self.prediction_length if self.prediction_length is not None else len(future_full)

        full_seq = np.nan_to_num(np.concatenate([history_full, future_full]), nan=0.0)
        future_full_clean = np.nan_to_num(future_full, nan=0.0)

        if offset == -1:
            ctx_end   = len(full_seq) - len(future_full)
            ctx_start = max(0, ctx_end - self.context_length)
            history   = full_seq[ctx_start:ctx_end]
            future    = future_full_clean[:pred_len]
        else:
            ctx_start = offset
            ctx_end   = ctx_start + self.context_length
            history   = full_seq[ctx_start:ctx_end]
            future    = full_seq[ctx_end:ctx_end + pred_len]

        if len(history) < self.context_length:
            pad = np.zeros(self.context_length, dtype=np.float32)
            pad[-len(history):] = history
            history = pad

        if len(future) < pred_len:
            pad = np.zeros(pred_len, dtype=np.float32)
            pad[:len(future)] = future
            future = pad

        # FD mode: shift context to include future (encoder upper-bound).
        if self.fd_mode:
            fd_seq    = np.concatenate([history_full, future])
            ctx_end   = len(fd_seq)
            ctx_start = max(0, ctx_end - self.context_length)
            history   = fd_seq[ctx_start:ctx_end]

        # Per-series z-score normalisation.
        # If history_full is entirely NaN (e.g. car_parts trailing-NaN series),
        # nan_to_num already zeroed it — sigma will be 0, so return zero tensors.
        history_full_clean = history_full[~np.isnan(history_full)]
        if len(history_full_clean) == 0:
            x = torch.zeros(self.context_length, 1, dtype=self.dtype)
            y = torch.zeros(pred_len,            1, dtype=self.dtype)
            return {
                "target":  x,
                "future":  y,
                "mu":      torch.tensor(0., dtype=torch.float32),
                "sigma":   torch.tensor(1., dtype=torch.float32),
                "item_id": item.get("item_id", "unknown"),
            }

        # Use full-history statistics when context window is unrepresentative:
        # (a) near-constant context (sigma_ctx < 1e-3), or
        # (b) context std < 50% of full-history std — low-variance regime that
        #     causes catastrophically large normalised futures in bizitobs_service.
        mu_ctx     = history.mean()
        sigma_ctx  = history.std()
        mu_hist    = history_full_clean.mean()
        sigma_hist = history_full_clean.std()
        if sigma_ctx < 1e-3 or (sigma_hist > 1e-8 and sigma_ctx / sigma_hist < 0.5):
            mu    = mu_hist
            sigma = float(sigma_hist) + 1e-8
        else:
            mu    = mu_ctx
            sigma = float(sigma_ctx) + 1e-8

        history = (history - mu) / sigma
        future  = (future  - mu) / sigma

        x = torch.as_tensor(history, dtype=self.dtype).unsqueeze(-1)  # (L, 1)
        y = torch.as_tensor(future,  dtype=self.dtype).unsqueeze(-1)  # (H, 1)

        return {
            "target":  x,
            "future":  y,
            "mu":      torch.tensor(mu,    dtype=torch.float32),
            "sigma":   torch.tensor(sigma, dtype=torch.float32),
            "item_id": item.get("item_id", "unknown"),
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
    prediction_length: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    repo_id: str = GIFT_EVAL_REPO_ID,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    force_offline: bool = False,
    fd_mode: bool = False,
    stride: Optional[int] = None,
    sliding_window: bool = False,
) -> DataLoader:
    """Build a DataLoader for a GIFT-Eval subset.

    prediction_length=None (default): use the native benchmark horizon from
    future_value, consistent with the official GIFT-Eval evaluation protocol.

    fd_mode=True: context window is shifted to include the future values.
    Useful as an encoder upper-bound: model sees the future during encoding but still predicts it.
    """
    ds_hf = load_gift_eval_hf(
        subset_name,
        repo_id=repo_id,
        cache_dir=cache_dir,
        force_offline=force_offline
    )

    ds_torch = GiftEvalDataset(
        ds_hf,
        context_length=context_length,
        prediction_length=prediction_length,
        fd_mode=fd_mode,
        stride=stride,
        sliding_window=sliding_window,
    )
    
    return DataLoader(
        ds_torch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    import warnings
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    import torch

    print(f"{'Subset':<40} {'#rows':>6}  {'target':>12}  {'future':>12}  {'NaN?'}")
    print("-" * 80)
    ok = err = 0
    for subset in ALL_GIFT_SSL_SUBSETS:
        try:
            loader = build_gift_eval_dataloader(
                subset, context_length=336, batch_size=16,
                num_workers=0, force_offline=True,
            )
            batch = next(iter(loader))
            has_nan = torch.isnan(batch["target"]).any() or torch.isnan(batch["future"]).any()
            tgt_s = str(tuple(batch["target"].shape))
            fut_s = str(tuple(batch["future"].shape))
            print(f"{subset:<40} {len(loader.dataset):>6}  {tgt_s:>12}  {fut_s:>12}  {'NaN!!!' if has_nan else 'ok'}")
            ok += 1
        except Exception as e:
            print(f"{subset:<40}  ERROR: {e}")
            err += 1
    print(f"\n{ok} passed, {err} failed.")
