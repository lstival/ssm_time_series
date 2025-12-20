"""Chronos dataset utilities for building PyTorch dataloaders."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import datasets
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Allow running this file directly: `python src/dataloaders/cronos_loader.py`
# In that case, relative imports (from .utils ...) fail because there's no parent package.
if __package__ in (None, ""):
	# Add `.../src` to sys.path so `import dataloaders.*` works.
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloaders.utils import ensure_hf_list_feature_registered
from dataloaders.cronos_dataset import load_chronos_datasets, target_only_view

logger = logging.getLogger(__name__)


# Keep backward compatibility with older Hugging Face `datasets` versions.
ensure_hf_list_feature_registered()

SplitSpec = Union[int, float]


class ChronosTorchDataset(Dataset):
	"""Minimal wrapper that exposes Chronos samples as torch tensors."""

	def __init__(
		self,
		hf_dataset: datasets.Dataset,
		*,
		torch_dtype: torch.dtype = torch.float32,
		normalization: Optional[Dict[str, Any]] = None,
	) -> None:
		self.dataset = target_only_view(hf_dataset, keep_format=True)
		self.dtype = torch_dtype
		self.normalization = normalization

	def __len__(self) -> int:  # pragma: no cover - simple passthrough
		return len(self.dataset)

	def __getitem__(self, index: int) -> torch.Tensor:
		sample = self.dataset[index]
		target = sample["target"]
		if isinstance(target, torch.Tensor):
			tensor = target.to(self.dtype)
		else:
			array = np.asarray(target)
			tensor = torch.as_tensor(array, dtype=self.dtype)

		if self.normalization is None:
			return tensor

		mode = str(self.normalization.get("mode") or "").lower()
		eps = float(self.normalization.get("epsilon", 1e-12))
		if mode == "global_minmax":
			dmin = float(self.normalization.get("min", 0.0))
			dmax = float(self.normalization.get("max", 0.0))
			rng = dmax - dmin
			denom = rng if abs(rng) > eps else 1.0
			return (tensor - dmin) / denom
		if mode == "global_standard":
			mean = float(self.normalization.get("mean", 0.0))
			std = float(self.normalization.get("std", 1.0))
			denom = std if abs(std) > eps else 1.0
			return (tensor - mean) / denom
		return tensor


def _compute_global_normalization(hf_dataset: datasets.Dataset, *, mode: str) -> Optional[Dict[str, Any]]:
	mode = str(mode or "none").strip().lower()
	if mode in {"", "none", "false", "0"}:
		return None

	# Flatten all values to compute stats. We keep this simple and robust; Chronos targets are 1D.
	values = np.asarray(hf_dataset["target"], dtype=object)
	flat: list[np.ndarray] = []
	for item in values:
		arr = np.asarray(item, dtype=np.float64).reshape(-1)
		if arr.size:
			flat.append(arr[np.isfinite(arr)])
	if not flat:
		return None
	concat = np.concatenate(flat, axis=0)
	concat = concat[np.isfinite(concat)]
	if concat.size == 0:
		return None

	if mode in {"global_minmax", "minmax"}:
		return {
			"mode": "global_minmax",
			"min": float(np.min(concat)),
			"max": float(np.max(concat)),
			"epsilon": 1e-12,
		}
	if mode in {"global_standard", "standard", "zscore"}:
		return {
			"mode": "global_standard",
			"mean": float(np.mean(concat)),
			"std": float(np.std(concat)),
			"epsilon": 1e-12,
		}
	return None


def chronos_collate_fn(
	batch: Sequence[torch.Tensor],
	*,
	pad_value: float = 0.0,
	return_lengths: bool = True,
	return_mask: bool = True,
) -> Dict[str, torch.Tensor]:
	if not batch:
		raise ValueError("Received an empty batch.")

	flat_batch = [sample.reshape(-1) for sample in batch]
	padded = pad_sequence(flat_batch, batch_first=True, padding_value=pad_value)

	result: Dict[str, torch.Tensor] = {"target": padded}

	if return_lengths or return_mask:
		# create lengths on the same device as padded
		lengths = torch.tensor([s.numel() for s in flat_batch], dtype=torch.long, device=padded.device)
		if return_lengths:
			result["lengths"] = lengths
		if return_mask:
			steps = torch.arange(padded.size(1), device=padded.device)
			mask = steps.unsqueeze(0) < lengths.unsqueeze(1)
			result["mask"] = mask

	return result


def _resolve_val_split(split: SplitSpec, total: int) -> SplitSpec:
	if isinstance(split, float):
		if not 0.0 < split < 1.0:
			raise ValueError("Float validation split must be in the open interval (0, 1).")
		if max(1, int(total * split)) == 0:
			raise ValueError("Validation split yields zero samples; increase split size or dataset.")
	else:
		if split <= 0 or split >= total:
			raise ValueError("Integer validation split must be between 1 and len(dataset) - 1.")
	return split


def build_chronos_dataloaders(
	datasets_to_load: Sequence[str],
	*,
	val_split: SplitSpec = 0.2,
	batch_size: int = 64,
	val_batch_size: Optional[int] = None,
	num_workers: int = 0,
	pin_memory: bool = False,
	shuffle_train: bool = True,
	shuffle_val: bool = False,
	drop_last: bool = False,
	pad_value: float = 0.0,
	torch_dtype: torch.dtype = torch.float32,
	repo_id: str = "autogluon/chronos_datasets",
	target_dtype: Optional[str] = "float32",
	normalize_mode: Optional[str] = None,
	seed: int = 42,
	load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader]:
	"""Create train/validation dataloaders sourced from Chronos datasets."""

	if not datasets_to_load:
		raise ValueError("At least one dataset name must be provided.")

	mode = str(normalize_mode or "per_series").strip().lower()
	normalize_per_series = mode in {"per_series", "series", "default"}

	merged = load_chronos_datasets(
		datasets_to_load,
		repo_id=repo_id,
		set_numpy_format=True,
		target_dtype=target_dtype,
		normalize_per_series=normalize_per_series,
		**(load_kwargs or {}),
	)

	total_records = len(merged)
	_resolve_val_split(val_split, total_records)

	split = merged.train_test_split(test_size=val_split, shuffle=True, seed=seed)
	train_hf = split["train"]
	val_hf = split["test"]

	normalization = None
	if not normalize_per_series:
		normalization = _compute_global_normalization(train_hf, mode=mode)

	train_dataset = ChronosTorchDataset(train_hf, torch_dtype=torch_dtype, normalization=normalization)
	val_dataset = ChronosTorchDataset(val_hf, torch_dtype=torch_dtype, normalization=normalization)

	collate = partial(
		chronos_collate_fn,
		pad_value=pad_value,
		return_lengths=True,
		return_mask=True,
	)

	val_batch = val_batch_size or batch_size

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle_train,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=drop_last,
		collate_fn=collate,
	)

	val_loader = DataLoader(
		val_dataset,
		batch_size=val_batch,
		shuffle=shuffle_val,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=False,
		collate_fn=collate,
	)

	logger.info(
		"Chronos dataloaders ready: %d train samples, %d val samples.",
		len(train_dataset),
		len(val_dataset),
	)

	return train_loader, val_loader


def build_chronos_dataloaders_by_dataset(
	datasets_to_load: Sequence[str],
	*,
	val_split: SplitSpec = 0.2,
	batch_size: int = 64,
	val_batch_size: Optional[int] = None,
	num_workers: int = 0,
	pin_memory: bool = False,
	shuffle_train: bool = True,
	shuffle_val: bool = False,
	drop_last: bool = False,
	pad_value: float = 0.0,
	torch_dtype: torch.dtype = torch.float32,
	repo_id: str = "autogluon/chronos_datasets",
	target_dtype: Optional[str] = "float32",
	normalize_mode: Optional[str] = None,
	seed: int = 42,
	load_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
	"""Create train/val dataloaders per dataset (no concatenation).

	This is useful when you want to avoid mixing datasets with incompatible
	time steps / semantics.
	"""
	if not datasets_to_load:
		raise ValueError("At least one dataset name must be provided.")

	val_batch = val_batch_size or batch_size
	collate = partial(
		chronos_collate_fn,
		pad_value=pad_value,
		return_lengths=True,
		return_mask=True,
	)

	result: Dict[str, Tuple[DataLoader, DataLoader]] = {}
	for name in datasets_to_load:
		mode = str(normalize_mode or "per_series").strip().lower()
		normalize_per_series = mode in {"per_series", "series", "default"}
		merged = load_chronos_datasets(
			[name],
			repo_id=repo_id,
			set_numpy_format=True,
			target_dtype=target_dtype,
			normalize_per_series=normalize_per_series,
			**(load_kwargs or {}),
		)
		total_records = len(merged)
		_resolve_val_split(val_split, total_records)
		split = merged.train_test_split(test_size=val_split, shuffle=True, seed=seed)
		train_hf = split["train"]
		val_hf = split["test"]

		normalization = None
		if not normalize_per_series:
			normalization = _compute_global_normalization(train_hf, mode=mode)

		train_dataset = ChronosTorchDataset(train_hf, torch_dtype=torch_dtype, normalization=normalization)
		val_dataset = ChronosTorchDataset(val_hf, torch_dtype=torch_dtype, normalization=normalization)

		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=shuffle_train,
			num_workers=num_workers,
			pin_memory=pin_memory,
			drop_last=drop_last,
			collate_fn=collate,
		)
		val_loader = DataLoader(
			val_dataset,
			batch_size=val_batch,
			shuffle=shuffle_val,
			num_workers=num_workers,
			pin_memory=pin_memory,
			drop_last=False,
			collate_fn=collate,
		)
		result[str(name)] = (train_loader, val_loader)

	return result


__all__ = [
	"ChronosTorchDataset",
	"build_chronos_dataloaders",
	"build_chronos_dataloaders_by_dataset",
	"chronos_collate_fn",
]

if __name__ == "__main__":
	import matplotlib.pyplot as plt 
	logging.basicConfig(level=logging.INFO)
	datasets_to_load = [
        "m4_daily",
        # "m4_hourly",
        # "m4_monthly",
        # "m4_yearly",
        # "monash_australian_electricity",
        # "taxi_30min",
        # "monash_traffic",
        # "monash_kdd_cup_2018",
        # "m5",
        # "mexico_city_bikes",
        # "exchange_rate",
        # "monash_car_parts",
        # "monash_covid_deaths",
        # "monash_electricity_hourly",
        # "monash_fred_md",
        # "monash_hospital",
        # "monash_m1_monthly",
        # "monash_m1_quarterly",
        # "monash_m1_yearly",
        # "monash_m3_monthly",
        # "monash_m3_quarterly",
        # "monash_m3_yearly",
        # "monash_nn5_weekly",
        "taxi_30min",
        # "uber_tlc_daily",
        # "uber_tlc_hourly",
        # "wind_farms_hourly",
        # "wind_farms_daily",
        # "dominick",
        # "electricity_15min",
        # "solar_1h",
        # "ercot", # 8 rows with 158k length time series
    ]

    # Build dataloaders (will download dataset if needed)
	train_loader, val_loader = build_chronos_dataloaders(
        datasets_to_load,
        val_split=0.2,
        batch_size=8,
        num_workers=0,
        seed=42,
		shuffle_train=False,
    )

    # Inspect a single batch from train loader
	batch = next(iter(train_loader))
	print("target shape:", batch["target"].shape)
	print("lengths:", batch.get("lengths"))
	print("mask shape:", batch.get("mask").shape)
	targets = batch["target"]
	lengths = batch.get("lengths")
	if lengths is None:
		lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)
	lengths = lengths.detach().cpu().numpy()

	# Inverse-transform back to original scale for plotting.
	# Chronos targets are normalized per-series in `load_chronos_datasets()` by default.
	# For visualization we reload the *raw* split (no normalization) and use its min/max.
	raw_merged = load_chronos_datasets(
		datasets_to_load,
		repo_id="autogluon/chronos_datasets",
		set_numpy_format=True,
		target_dtype="float32",
		normalize_per_series=False,
	)
	raw_split = raw_merged.train_test_split(test_size=0.2, shuffle=True, seed=42)
	raw_train = raw_split["train"]

	denorm_targets = []
	for i in range(targets.shape[0]):
		valid_len = int(lengths[i])
		norm = targets[i, :valid_len].detach().cpu().reshape(-1).numpy()
		raw = np.asarray(raw_train[i]["target"]).reshape(-1)
		seq_min = float(np.nanmin(raw)) if raw.size else 0.0
		seq_max = float(np.nanmax(raw)) if raw.size else 0.0
		range_val = seq_max - seq_min
		if abs(range_val) < 1e-12:
			denorm = np.full_like(norm, seq_min, dtype=np.float32)
		else:
			denorm = (norm * range_val + seq_min).astype(np.float32, copy=False)
		denorm_targets.append(denorm)
	rows, cols = 4, 2
	n_plots = min(targets.shape[0], rows * cols)
	fig, axes = plt.subplots(rows, cols, figsize=(targets.shape[0], 12))
	axes = axes.flatten()
	for i in range(n_plots):
		arr = denorm_targets[i]
		ax = axes[i]
		ax.plot(arr, alpha=0.8)
		ax.set_title(f"sample {i} (len={arr.size})")
		ax.set_xlabel("time step")
		ax.set_ylabel("value")
	for j in range(n_plots, rows * cols):
		axes[j].axis("off")
	plt.tight_layout()
	plt.show()
