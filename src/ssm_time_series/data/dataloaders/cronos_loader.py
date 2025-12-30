"""Chronos dataset utilities for building PyTorch dataloaders."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import datasets
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from cronos_dataset import load_chronos_datasets, target_only_view
except:
	from .cronos_dataset import load_chronos_datasets, target_only_view

logger = logging.getLogger(__name__)

SplitSpec = Union[int, float]


class ChronosTorchDataset(Dataset):
	"""Minimal wrapper that exposes Chronos samples as torch tensors."""

	def __init__(
		self,
		hf_dataset: datasets.Dataset,
		*,
		torch_dtype: torch.dtype = torch.float32,
	) -> None:
		self.dataset = target_only_view(hf_dataset, keep_format=True)
		self.dtype = torch_dtype

	def __len__(self) -> int:  # pragma: no cover - simple passthrough
		return len(self.dataset)

	def __getitem__(self, index: int) -> torch.Tensor:
		sample = self.dataset[index]
		target = sample["target"]
		if isinstance(target, torch.Tensor):
			return target.to(self.dtype)
		array = np.asarray(target)
		return torch.as_tensor(array, dtype=self.dtype)


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
	seed: int = 42,
	load_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader]:
	"""Create train/validation dataloaders sourced from Chronos datasets."""

	if not datasets_to_load:
		raise ValueError("At least one dataset name must be provided.")

	merged = load_chronos_datasets(
		datasets_to_load,
		repo_id=repo_id,
		set_numpy_format=True,
		target_dtype=target_dtype,
		**(load_kwargs or {}),
	)

	total_records = len(merged)
	_resolve_val_split(val_split, total_records)

	split = merged.train_test_split(test_size=val_split, shuffle=True, seed=seed)
	train_hf = split["train"]
	val_hf = split["test"]

	train_dataset = ChronosTorchDataset(train_hf, torch_dtype=torch_dtype)
	val_dataset = ChronosTorchDataset(val_hf, torch_dtype=torch_dtype)

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


__all__ = [
	"ChronosTorchDataset",
	"build_chronos_dataloaders",
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
        # "taxi_30min",
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
	rows, cols = 4, 2
	n_plots = min(targets.shape[0], rows * cols)
	fig, axes = plt.subplots(rows, cols, figsize=(targets.shape[0], 12))
	axes = axes.flatten()
	for i in range(n_plots):
		s = targets[i]  # torch.Tensor
		arr = s.detach().cpu().reshape(-1).numpy()
		valid_len = int(lengths[i])
		arr = arr[:valid_len]  # trim to actual length before plotting
		ax = axes[i]
		ax.plot(arr, alpha=0.8)
		ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
		ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
		ax.set_title(f"sample {i} (len={arr.size})")
		ax.set_xlabel("time step")
		ax.set_ylabel("value")
	for j in range(n_plots, rows * cols):
		axes[j].axis("off")
	plt.tight_layout()
	plt.show()
