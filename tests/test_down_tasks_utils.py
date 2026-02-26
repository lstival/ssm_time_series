"""Tests for down_tasks helper utilities.

Covers:
  - parse_horizon_values: happy-path parsing, deduplication, sorting, error cases
  - apply_model_overrides: selective override, defaults, original dict unchanged
  - ensure_dataloader_pred_len: propagates pred_len to datasets, None-safety,
    ConcatDataset traversal
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

from down_tasks.forecast_shared import parse_horizon_values, apply_model_overrides
from down_tasks.forecast_utils import ensure_dataloader_pred_len


# ==========================================================================
# parse_horizon_values
# ==========================================================================

class TestParseHorizonValues:
    def test_basic_csv(self):
        result = parse_horizon_values("24,48,96")
        assert result == [24, 48, 96]

    def test_single_value(self):
        result = parse_horizon_values("168")
        assert result == [168]

    def test_duplicates_are_removed(self):
        result = parse_horizon_values("24,24,48")
        assert result == [24, 48]

    def test_output_is_sorted(self):
        result = parse_horizon_values("96,24,48")
        assert result == sorted(result)

    def test_whitespace_around_values(self):
        result = parse_horizon_values(" 24 , 48 , 96 ")
        assert result == [24, 48, 96]

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_horizon_values("")

    def test_only_commas_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_horizon_values(",,,")

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            parse_horizon_values("0,24")

    def test_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            parse_horizon_values("-1,24")

    def test_non_integer_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_horizon_values("24,abc,96")

    def test_float_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_horizon_values("24.5,48")

    def test_returns_list_of_ints(self):
        result = parse_horizon_values("12,24,48")
        assert all(isinstance(h, int) for h in result)

    def test_large_horizons(self):
        result = parse_horizon_values("336,720")
        assert result == [336, 720]


# ==========================================================================
# apply_model_overrides
# ==========================================================================

class TestApplyModelOverrides:
    @pytest.fixture
    def base_cfg(self):
        return {
            "input_dim": 128,
            "model_dim": 256,
            "embedding_dim": 64,
            "depth": 2,
        }

    def test_does_not_mutate_original(self, base_cfg):
        original_input_dim = base_cfg["input_dim"]
        apply_model_overrides(base_cfg, token_size=99)
        assert base_cfg["input_dim"] == original_input_dim

    def test_returns_new_dict(self, base_cfg):
        result = apply_model_overrides(base_cfg)
        assert result is not base_cfg

    def test_token_size_overrides_input_dim(self, base_cfg):
        result = apply_model_overrides(base_cfg, token_size=32)
        assert result["input_dim"] == 32

    def test_model_dim_override(self, base_cfg):
        result = apply_model_overrides(base_cfg, model_dim=512)
        assert result["model_dim"] == 512

    def test_embedding_dim_override(self, base_cfg):
        result = apply_model_overrides(base_cfg, embedding_dim=128)
        assert result["embedding_dim"] == 128

    def test_depth_override(self, base_cfg):
        result = apply_model_overrides(base_cfg, depth=4)
        assert result["depth"] == 4

    def test_none_override_leaves_original_value(self, base_cfg):
        result = apply_model_overrides(base_cfg, model_dim=None)
        assert result["model_dim"] == base_cfg["model_dim"]

    def test_default_pooling_added_when_absent(self, base_cfg):
        result = apply_model_overrides(base_cfg)
        assert result["pooling"] == "mean"

    def test_default_dropout_added_when_absent(self, base_cfg):
        result = apply_model_overrides(base_cfg)
        assert result["dropout"] == pytest.approx(0.1)

    def test_existing_pooling_not_overwritten(self, base_cfg):
        base_cfg["pooling"] = "last"
        result = apply_model_overrides(base_cfg)
        assert result["pooling"] == "last"

    def test_existing_dropout_not_overwritten(self, base_cfg):
        base_cfg["dropout"] = 0.5
        result = apply_model_overrides(base_cfg)
        assert result["dropout"] == pytest.approx(0.5)

    def test_custom_default_pooling(self, base_cfg):
        result = apply_model_overrides(base_cfg, default_pooling="cls")
        assert result["pooling"] == "cls"

    def test_custom_default_dropout(self, base_cfg):
        result = apply_model_overrides(base_cfg, default_dropout=0.3)
        assert result["dropout"] == pytest.approx(0.3)

    def test_original_keys_preserved(self, base_cfg):
        result = apply_model_overrides(base_cfg, depth=3)
        # Keys not overridden must survive
        assert result["input_dim"] == base_cfg["input_dim"]
        assert result["model_dim"] == base_cfg["model_dim"]
        assert result["embedding_dim"] == base_cfg["embedding_dim"]

    def test_empty_base_config(self):
        result = apply_model_overrides({}, token_size=16, model_dim=32, embedding_dim=8, depth=1)
        assert result["input_dim"] == 16
        assert result["model_dim"] == 32
        assert result["embedding_dim"] == 8
        assert result["depth"] == 1


# ==========================================================================
# ensure_dataloader_pred_len
# ==========================================================================

class _SimplePredDataset:
    """Minimal dataset with a `pred_len` attribute."""
    def __init__(self, n: int = 8, pred_len: int = 24):
        self.data    = torch.zeros(n, 10)
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class TestEnsureDataloaderPredLen:
    def test_none_loader_is_safe(self):
        """Should not raise when loader is None."""
        ensure_dataloader_pred_len(None, 48)  # no assertion needed — just no exception

    def test_sets_pred_len_on_simple_dataset(self):
        ds     = _SimplePredDataset(pred_len=24)
        loader = DataLoader(ds, batch_size=2)
        ensure_dataloader_pred_len(loader, 96)
        assert ds.pred_len == 96

    def test_no_effect_on_dataset_without_pred_len(self):
        """TensorDataset has no `pred_len`; function must not crash."""
        x  = torch.randn(8, 10)
        ds = TensorDataset(x)
        loader = DataLoader(ds, batch_size=2)
        ensure_dataloader_pred_len(loader, 48)  # must not raise

    def test_sets_pred_len_on_concat_dataset(self):
        ds1 = _SimplePredDataset(n=4, pred_len=24)
        ds2 = _SimplePredDataset(n=4, pred_len=24)
        cat_ds = ConcatDataset([ds1, ds2])
        loader = DataLoader(cat_ds, batch_size=2)
        ensure_dataloader_pred_len(loader, 48)
        assert ds1.pred_len == 48
        assert ds2.pred_len == 48

    def test_multiple_calls_are_idempotent(self):
        ds     = _SimplePredDataset(pred_len=24)
        loader = DataLoader(ds, batch_size=2)
        ensure_dataloader_pred_len(loader, 96)
        ensure_dataloader_pred_len(loader, 96)
        assert ds.pred_len == 96

    def test_pred_len_updated_to_new_value(self):
        ds     = _SimplePredDataset(pred_len=24)
        loader = DataLoader(ds, batch_size=2)
        ensure_dataloader_pred_len(loader, 192)
        assert ds.pred_len == 192

    def test_deeply_wrapped_dataset(self):
        """Handles DataLoader → Subset-like wrapper with .dataset attribute."""
        from torch.utils.data import Subset
        ds     = _SimplePredDataset(pred_len=24)
        subset = Subset(ds, list(range(4)))
        loader = DataLoader(subset, batch_size=2)
        ensure_dataloader_pred_len(loader, 72)
        assert ds.pred_len == 72
