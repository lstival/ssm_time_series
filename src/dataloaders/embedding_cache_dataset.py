"""Dataset utilities for encoder embedding caches created by extract_icml_embeddings."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class _BatchInfo:
    path: Path
    length: int
    end_index: int


class EmbeddingCacheDataset(Dataset):
    """Dataset that streams cached encoder embeddings from disk.

    Parameters
    ----------
    root : str or Path
        Root directory produced by ``extract_icml_embeddings`` for a dataset.
    split : str
        Which split to load (``train``, ``val``, or ``test``).
    horizon : int
        Forecast horizon key to retrieve (e.g. 96, 192, 336, 720).
    transform : callable, optional
        Optional transform applied to the embedding tensor.
    target_transform : callable, optional
        Optional transform applied to the target tensor.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        horizon: int = 96,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.horizon = int(horizon)
        self.transform = transform
        self.target_transform = target_transform
        self._target_key = f"targets_{self.horizon}"

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        batch_files = sorted(path for path in split_dir.glob("batch_*.pt") if path.is_file())
        if not batch_files:
            raise FileNotFoundError(f"No cached batches found under {split_dir}")

        self._batches: List[_BatchInfo] = []
        self._total_samples = 0
        self._embedding_shape: Optional[Tuple[int, ...]] = None
        self._target_shape: Optional[Tuple[int, ...]] = None
        self._cache_path: Optional[Path] = None
        self._cache_payload: Optional[Dict[str, Tensor]] = None

        for file_path in batch_files:
            payload = torch.load(file_path, map_location="cpu")
            if "embeddings" not in payload or self._target_key not in payload:
                raise KeyError(f"File {file_path} missing required keys 'embeddings' or '{self._target_key}'")
            embeddings = payload["embeddings"]
            targets = payload[self._target_key]
            if embeddings.ndim < 2:
                raise ValueError(f"Embeddings in {file_path} must be at least 2D, got {embeddings.shape}")
            if targets.size(0) != embeddings.size(0):
                raise ValueError(f"Mismatched batch size in {file_path}: embeddings {embeddings.size(0)} vs targets {targets.size(0)}")

            batch_length = int(embeddings.size(0))
            self._total_samples += batch_length
            self._batches.append(_BatchInfo(path=file_path, length=batch_length, end_index=self._total_samples))

            if self._embedding_shape is None:
                self._embedding_shape = tuple(int(dim) for dim in embeddings.shape[1:])
            if self._target_shape is None:
                self._target_shape = tuple(int(dim) for dim in targets.shape[1:])

            # Drop strong references to save RAM when many batches exist.
            del payload

        if self._embedding_shape is None or self._target_shape is None:
            raise RuntimeError("Failed to infer embedding/target shapes from cache")

    def __len__(self) -> int:  # noqa: D401 - torch Dataset requirement
        return self._total_samples

    def _load_batch(self, record: _BatchInfo) -> Dict[str, Tensor]:
        if self._cache_path != record.path:
            payload = torch.load(record.path, map_location="cpu")
            self._cache_path = record.path
            self._cache_payload = payload
        assert self._cache_payload is not None  # narrow type checker
        return self._cache_payload

    def _locate(self, index: int) -> Tuple[_BatchInfo, int]:
        if index < 0:
            index += self._total_samples
        if index < 0 or index >= self._total_samples:
            raise IndexError(f"Index {index} out of range for dataset with {self._total_samples} samples")
        batch_idx = bisect_left([record.end_index for record in self._batches], index + 1)
        record = self._batches[batch_idx]
        prev_end = 0 if batch_idx == 0 else self._batches[batch_idx - 1].end_index
        local_index = index - prev_end
        return record, local_index

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        record, local_index = self._locate(index)
        payload = self._load_batch(record)
        embeddings = payload["embeddings"][local_index]
        targets = payload[self._target_key][local_index]

        if self.transform is not None:
            embeddings = self.transform(embeddings)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return embeddings, targets

    @property
    def embedding_shape(self) -> Tuple[int, ...]:
        return self._embedding_shape  # type: ignore[return-value]

    @property
    def target_shape(self) -> Tuple[int, ...]:
        return self._target_shape  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"EmbeddingCacheDataset(root={self.root!s}, split={self.split!r}, horizon={self.horizon}, "
            f"samples={self._total_samples})"
        )


def build_embedding_cache_loader(
    root: str | Path,
    *,
    horizon: int,
    split: str = "train",
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    transform: Optional[Callable[[Tensor], Tensor]] = None,
    target_transform: Optional[Callable[[Tensor], Tensor]] = None,
) -> DataLoader:
    """Factory that creates a ``DataLoader`` over the cached embeddings."""

    dataset = EmbeddingCacheDataset(
        root,
        split=split,
        horizon=horizon,
        transform=transform,
        target_transform=target_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
