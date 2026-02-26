"""Tests for dataloaders.utils helpers.

Covers:
  - discover_dataset_files: extension filter, filename filter, relative paths
  - split_dataset: ratio edge-cases (0, 1, 0.5), empty dataset
  - _split_sequences: ratio, determinism, seed independence, empty input
  - _sequence_to_array: shape promotions (0-D, 1-D, 2-D, 3-D)
  - ChronosDatasetGroup dataclass creation
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from dataloaders.utils import (
    discover_dataset_files,
    split_dataset,
    _split_sequences,
    _sequence_to_array,
    ChronosDatasetGroup,
)


# ==========================================================================
# discover_dataset_files
# ==========================================================================

class TestDiscoverDatasetFiles:
    @pytest.fixture()
    def tmp_dir(self, tmp_path):
        """Create a small temp directory tree with mixed file types."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        (tmp_path / "series.txt").write_text("1\n2\n")
        (tmp_path / "sub" / "nested.npz").write_bytes(b"")
        (tmp_path / "README.md").write_text("doc")
        return tmp_path

    def test_finds_all_supported_extensions(self, tmp_dir):
        result = discover_dataset_files(str(tmp_dir))
        names = {os.path.basename(v) for v in result.values()}
        assert "data.csv" in names
        assert "series.txt" in names
        assert "nested.npz" in names
        assert "README.md" not in names

    def test_extension_filter(self, tmp_dir):
        result = discover_dataset_files(str(tmp_dir), extensions=[".csv"])
        assert all(v.endswith(".csv") for v in result.values())

    def test_filename_filter_returns_one(self, tmp_dir):
        result = discover_dataset_files(str(tmp_dir), filename="data.csv")
        assert len(result) == 1
        assert list(result.values())[0].endswith("data.csv")

    def test_empty_dir_returns_empty(self, tmp_path):
        result = discover_dataset_files(str(tmp_path))
        assert result == {}

    def test_nonexistent_filename_returns_empty(self, tmp_dir):
        result = discover_dataset_files(str(tmp_dir), filename="ghost.csv")
        assert result == {}

    def test_values_are_absolute_paths(self, tmp_dir):
        result = discover_dataset_files(str(tmp_dir))
        for v in result.values():
            assert os.path.isabs(v)


# ==========================================================================
# split_dataset
# ==========================================================================

class TestSplitDataset:
    @pytest.fixture()
    def ds(self):
        return TensorDataset(torch.arange(100, dtype=torch.float32))

    def test_50_50_split(self, ds):
        train, val = split_dataset(ds, train_ratio=0.5)
        assert len(train) == 50
        assert len(val) == 50

    def test_ratio_1_returns_all_train(self, ds):
        train, val = split_dataset(ds, train_ratio=1.0)
        assert len(train) == 100
        assert val is None

    def test_ratio_0_returns_all_val(self, ds):
        train, val = split_dataset(ds, train_ratio=0.0)
        # split_idx = 0 → train is None, val is everything
        assert train is None
        assert len(val) == 100

    def test_80_20_split_sizes(self, ds):
        train, val = split_dataset(ds, train_ratio=0.8)
        assert len(train) == 80
        assert len(val) == 20

    def test_empty_dataset_returns_nones(self):
        ds = TensorDataset(torch.tensor([]))
        train, val = split_dataset(ds, train_ratio=0.8)
        assert train is None
        assert val is None

    def test_single_element_train_ratio_1(self):
        ds = TensorDataset(torch.tensor([1.0]))
        train, val = split_dataset(ds, train_ratio=1.0)
        assert len(train) == 1
        assert val is None

    def test_train_and_val_are_disjoint(self, ds):
        train, val = split_dataset(ds, train_ratio=0.7)
        train_indices = set(train.indices)
        val_indices = set(val.indices)
        assert train_indices.isdisjoint(val_indices)

    def test_union_covers_full_dataset(self, ds):
        train, val = split_dataset(ds, train_ratio=0.6)
        all_indices = sorted(list(train.indices) + list(val.indices))
        assert all_indices == list(range(100))


# ==========================================================================
# _split_sequences
# ==========================================================================

class TestSplitSequences:
    @pytest.fixture()
    def seqs(self) -> List[np.ndarray]:
        rng = np.random.default_rng(0)
        return [rng.standard_normal(50).astype(np.float32) for _ in range(20)]

    def test_sizes_match_ratio(self, seqs):
        train, val = _split_sequences(seqs, val_ratio=0.2, seed=42)
        assert len(val) == 4    # 20 * 0.2 = 4
        assert len(train) == 16

    def test_union_covers_all(self, seqs):
        train, val = _split_sequences(seqs, val_ratio=0.3, seed=0)
        assert len(train) + len(val) == len(seqs)

    def test_zero_val_ratio(self, seqs):
        train, val = _split_sequences(seqs, val_ratio=0.0, seed=0)
        assert len(val) == 0
        assert len(train) == len(seqs)

    def test_empty_input_returns_empty(self):
        train, val = _split_sequences([], val_ratio=0.2, seed=0)
        assert train == []
        assert val == []

    def test_determinism_same_seed(self, seqs):
        t1, v1 = _split_sequences(seqs, val_ratio=0.3, seed=99)
        t2, v2 = _split_sequences(seqs, val_ratio=0.3, seed=99)
        assert [a.tolist() for a in t1] == [b.tolist() for b in t2]
        assert [a.tolist() for a in v1] == [b.tolist() for b in v2]

    def test_different_seeds_differ(self, seqs):
        _, v1 = _split_sequences(seqs, val_ratio=0.5, seed=1)
        _, v2 = _split_sequences(seqs, val_ratio=0.5, seed=2)
        # Very unlikely to be identical unless list is trivial
        if len(seqs) > 2:
            assert not all(a.tolist() == b.tolist() for a, b in zip(v1, v2))


# ==========================================================================
# _sequence_to_array
# ==========================================================================

class TestSequenceToArray:
    def test_1d_input_becomes_2d(self):
        arr = np.array([1.0, 2.0, 3.0])
        out = _sequence_to_array(arr)
        assert out.ndim == 2
        assert out.shape == (3, 1)
        assert out.dtype == np.float32

    def test_2d_input_unchanged_shape(self):
        arr = np.ones((10, 3))
        out = _sequence_to_array(arr)
        assert out.shape == (10, 3)
        assert out.dtype == np.float32

    def test_scalar_becomes_2d(self):
        out = _sequence_to_array(np.float32(7.0))
        assert out.ndim == 2
        assert out.shape == (1, 1)

    def test_torch_tensor_input(self):
        t = torch.randn(8, 2)
        out = _sequence_to_array(t.numpy())
        assert out.shape == (8, 2)
        assert out.dtype == np.float32

    def test_3d_input_flattened_last_dims(self):
        arr = np.ones((5, 2, 3))   # 5 timesteps, 2×3 features
        out = _sequence_to_array(arr)
        assert out.shape == (5, 6)


# ==========================================================================
# ChronosDatasetGroup – dataclass sanity
# ==========================================================================

class TestChronosDatasetGroup:
    def test_creation(self):
        group = ChronosDatasetGroup(name="test_ds", train=None, val=None)
        assert group.name == "test_ds"
        assert group.train is None
        assert group.val is None
        assert group.metadata == {}

    def test_metadata_field(self):
        group = ChronosDatasetGroup(name="x", train=None, val=None, metadata={"split": "train"})
        assert group.metadata["split"] == "train"
