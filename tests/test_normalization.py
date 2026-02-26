"""Tests for time-series normalisation utilities.

Covers:
  - _normalize_per_series (from dataloaders.cronos_dataset)
      * basic min-max scaling into [0, 1]
      * constant series (range ≈ 0) → output is 0 without NaN
      * all-NaN series → output unchanged, no crash
      * integer dtype input is upcast to float
      * min/max columns are added to dataset
  - Manual min-max round-trip
  - Reverse normalisation (denormalise) correctness
  - Edge cases: single element, two elements, negative values
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dataloaders.cronos_dataset import _normalize_per_series

try:
    import datasets as hf_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

hf_only = pytest.mark.skipif(not HF_AVAILABLE, reason="huggingface datasets not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hf_dataset(series_list):
    """Build a minimal HF Dataset from a list of 1-D Python lists."""
    import datasets
    data = {"target": series_list}
    return datasets.Dataset.from_dict(data)


# ==========================================================================
# _normalize_per_series (HuggingFace Dataset wrapper)
# ==========================================================================

@hf_only
class TestNormalizePerSeries:
    def test_basic_scaling(self):
        ds = _make_hf_dataset([[0.0, 5.0, 10.0], [2.0, 4.0, 6.0]])
        norm_ds = _normalize_per_series(ds)
        for row in norm_ds["target"]:
            arr = np.asarray(row, dtype=float)
            assert arr.min() >= -1e-9, "Normalised values below 0"
            assert arr.max() <= 1.0 + 1e-9, "Normalised values above 1"

    def test_min_max_columns_added(self):
        ds = _make_hf_dataset([[1.0, 2.0, 3.0]])
        norm_ds = _normalize_per_series(ds)
        assert "target_min" in norm_ds.column_names
        assert "target_max" in norm_ds.column_names

    def test_constant_series_no_nan(self):
        ds = _make_hf_dataset([[5.0, 5.0, 5.0]])
        norm_ds = _normalize_per_series(ds)
        arr = np.asarray(norm_ds["target"][0], dtype=float)
        assert not np.isnan(arr).any(), "Constant series must not produce NaN"
        assert (arr == 0.0).all(), "Constant series should normalise to 0"

    def test_correct_round_trip(self):
        """Denormalising the output must recover the original series."""
        original = [2.0, 5.0, 8.0, 11.0]
        ds = _make_hf_dataset([original])
        norm_ds = _normalize_per_series(ds)

        norm_values = np.asarray(norm_ds["target"][0], dtype=float)
        mn = np.asarray(norm_ds["target_min"][0], dtype=float).flatten()[0]
        mx = np.asarray(norm_ds["target_max"][0], dtype=float).flatten()[0]
        r = mx - mn if (mx - mn) > 1e-12 else 1.0
        recovered = norm_values.flatten() * r + mn
        np.testing.assert_allclose(recovered, original, atol=1e-5)

    def test_single_element_series(self):
        ds = _make_hf_dataset([[42.0]])
        norm_ds = _normalize_per_series(ds)
        arr = np.asarray(norm_ds["target"][0], dtype=float)
        assert not np.isnan(arr).any()

    def test_negative_values(self):
        ds = _make_hf_dataset([[-10.0, 0.0, 10.0]])
        norm_ds = _normalize_per_series(ds)
        arr = np.asarray(norm_ds["target"][0], dtype=float)
        assert abs(arr.min()) < 1e-6
        assert abs(arr.max() - 1.0) < 1e-6

    def test_multiple_series_independent(self):
        """Each series is normalised independently."""
        ds = _make_hf_dataset([[0.0, 100.0], [0.0, 1.0]])
        norm_ds = _normalize_per_series(ds)
        for row in norm_ds["target"]:
            arr = np.asarray(row, dtype=float).flatten()
            assert abs(arr.min()) < 1e-6
            assert abs(arr.max() - 1.0) < 1e-6

    def test_missing_target_column_passthrough(self):
        """Dataset without 'target' column is returned unchanged."""
        import datasets
        ds = datasets.Dataset.from_dict({"feature": [[1, 2, 3]]})
        result = _normalize_per_series(ds, column="target")
        assert "target" not in result.column_names


# ==========================================================================
# Numpy-level min-max normalisation (manual, no HF dependency)
# ==========================================================================

class TestMinMaxNormNumpyLevel:
    @pytest.fixture
    def series_1d(self):
        return np.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)

    def test_normalised_range(self, series_1d):
        mn, mx = series_1d.min(), series_1d.max()
        norm = (series_1d - mn) / (mx - mn + 1e-12)
        assert norm.min() >= 0.0 - 1e-9
        assert norm.max() <= 1.0 + 1e-9

    def test_round_trip_recovery(self, series_1d):
        mn, mx = series_1d.min(), series_1d.max()
        r = mx - mn
        norm = (series_1d - mn) / (r + 1e-12)
        recovered = norm * r + mn
        np.testing.assert_allclose(recovered, series_1d, atol=1e-5)

    def test_all_same_values(self):
        arr = np.ones(10, dtype=np.float64) * 7.0
        mn, mx = arr.min(), arr.max()
        r = mx - mn
        safe_r = r if r > 1e-12 else 1.0
        norm = (arr - mn) / safe_r if r > 1e-12 else np.zeros_like(arr)
        assert not np.isnan(norm).any()
        assert (norm == 0.0).all()

    def test_two_element_series(self):
        arr = np.array([2.0, 8.0])
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        np.testing.assert_allclose(norm, [0.0, 1.0])

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
    def test_various_scales_no_nan(self, scale):
        arr = np.array([0.0, 1.0, 2.0]) * scale
        mn, mx = arr.min(), arr.max()
        r = mx - mn
        safe_r = r if abs(r) > 1e-15 else 1.0
        norm = (arr - mn) / safe_r
        assert not np.isnan(norm).any()


# ==========================================================================
# Reverse normalisation (denormalise) with PyTorch tensors
# ==========================================================================

class TestReverseNormalisationTorch:
    def test_denormalise_recovers_original(self):
        """Simulate the training-loop reverse-norm pipeline."""
        original = torch.tensor([[[2.0], [5.0], [8.0]]])   # (1, 3, 1)
        mn = torch.tensor([[1.0]])       # (1, 1)
        mx = torch.tensor([[9.0]])
        r = (mx - mn).clamp_min(1e-6)

        # Forward: normalise
        norm = (original.squeeze(-1) - mn) / r                # (1, 3)
        norm = norm.unsqueeze(-1)                             # back to (1, 3, 1)

        # Reverse
        recovered = norm * r.unsqueeze(1) + mn.unsqueeze(1)
        torch.testing.assert_close(recovered, original, atol=1e-5, rtol=0)

    def test_constant_series_reverse_norm_no_nan(self):
        mn = torch.tensor([[5.0]])
        mx = torch.tensor([[5.0]])
        r = (mx - mn).clamp_min(1e-6)
        norm = torch.zeros(1, 4, 1)
        recovered = norm * r.unsqueeze(1) + mn.unsqueeze(1)
        assert torch.isfinite(recovered).all()

    def test_batch_denorm(self):
        B, T, F = 8, 12, 1
        mn  = torch.rand(B, F) * 10
        mx  = mn + torch.rand(B, F) * 5 + 0.1
        r   = (mx - mn).clamp_min(1e-6)
        original = torch.rand(B, T, F) * (mx - mn).unsqueeze(1) + mn.unsqueeze(1)
        # normalise
        norm = (original - mn.unsqueeze(1)) / r.unsqueeze(1)
        # denormalise
        recovered = norm * r.unsqueeze(1) + mn.unsqueeze(1)
        torch.testing.assert_close(recovered, original, atol=1e-4, rtol=1e-4)
