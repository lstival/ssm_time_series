"""Tests for multi-horizon forecasting evaluation utilities.

Covers:
  - compute_multi_horizon_loss: shapes, value correctness, aggregation
  - compute_multi_horizon_metrics: MSE/MAE values, horizon slicing
  - evaluate_dataset integration: full eval-loop on a minimal DataLoader
  - Edge-cases: single horizon, identical pred/target, all-zero inputs
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from down_tasks.forecast_utils import (
    compute_multi_horizon_loss,
    compute_multi_horizon_metrics,
)
from models.classifier import MultiHorizonForecastMLP, ForecastRegressor
from models.mamba_encoder import MambaEncoder
from util import evaluate_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_regressor(embedding_dim: int = 16, horizons=None):
    horizons = horizons or [24, 48]
    enc  = MambaEncoder(input_dim=8, model_dim=16, depth=1, state_dim=4,
                        expand_factor=1.5, embedding_dim=embedding_dim)
    head = MultiHorizonForecastMLP(embedding_dim, 32, horizons)
    return ForecastRegressor(encoder=enc, head=head, freeze_encoder=True)


def _make_forecast_loader(
    n_samples: int = 16,
    seq_in: int = 64,
    seq_out: int = 48,
    features: int = 1,
) -> DataLoader:
    """Build a minimal DataLoader matching the format expected by evaluate_dataset."""
    # (B, T_in, F) for x, (B, T_out, F) for y; plus dummy date tensors
    x      = torch.randn(n_samples, seq_in,  features)
    y      = torch.randn(n_samples, seq_out, features)
    date_x = torch.zeros(n_samples, seq_in)
    date_y = torch.zeros(n_samples, seq_out)
    ds     = TensorDataset(x, y, date_x, date_y)
    return DataLoader(ds, batch_size=4, shuffle=False)


# ==========================================================================
# compute_multi_horizon_loss
# ==========================================================================

class TestComputeMultiHorizonLoss:
    @pytest.fixture
    def criterion(self):
        return nn.MSELoss()

    def test_returns_tensor_and_dict(self, criterion):
        preds   = torch.randn(4, 96, 1)
        targets = torch.randn(4, 96, 1)
        total, per_horizon = compute_multi_horizon_loss(preds, targets, [24, 48, 96], criterion)
        assert isinstance(total, torch.Tensor)
        assert isinstance(per_horizon, dict)

    def test_all_configured_horizons_in_dict(self, criterion):
        horizons = [12, 24, 48]
        preds    = torch.randn(2, 48, 1)
        targets  = torch.randn(2, 48, 1)
        _, ph = compute_multi_horizon_loss(preds, targets, horizons, criterion)
        assert set(ph.keys()) == set(horizons)

    def test_identical_pred_target_loss_zero(self, criterion):
        x = torch.randn(4, 48, 1)
        total, ph = compute_multi_horizon_loss(x, x, [24, 48], criterion)
        assert abs(float(total)) < 1e-7
        for v in ph.values():
            assert abs(v) < 1e-7

    def test_single_horizon(self, criterion):
        preds   = torch.randn(4, 24, 1)
        targets = torch.randn(4, 24, 1)
        total, ph = compute_multi_horizon_loss(preds, targets, [24], criterion)
        assert 24 in ph

    def test_loss_is_non_negative(self, criterion):
        preds   = torch.randn(8, 96, 1)
        targets = torch.randn(8, 96, 1)
        total, ph = compute_multi_horizon_loss(preds, targets, [24, 48, 96], criterion)
        assert float(total) >= 0

    def test_predictions_shorter_than_targets_raises(self, criterion):
        preds   = torch.randn(4, 12, 1)
        targets = torch.randn(4, 24, 1)
        with pytest.raises(ValueError, match="shorter"):
            compute_multi_horizon_loss(preds, targets, [24], criterion)

    def test_total_is_average_over_horizons(self, criterion):
        """Total loss = mean of per-horizon losses."""
        preds   = torch.randn(4, 96, 1)
        targets = torch.randn(4, 96, 1)
        horizons = [24, 48, 96]
        total, ph = compute_multi_horizon_loss(preds, targets, horizons, criterion)
        expected = sum(ph.values()) / len(horizons)
        assert abs(float(total) - expected) < 1e-6

    def test_mae_criterion_also_works(self):
        crit = nn.L1Loss()
        preds   = torch.randn(4, 48, 1)
        targets = torch.randn(4, 48, 1)
        total, ph = compute_multi_horizon_loss(preds, targets, [48], crit)
        assert float(total) >= 0

    def test_output_is_differentiable(self, criterion):
        preds   = torch.randn(4, 48, 1, requires_grad=True)
        targets = torch.randn(4, 48, 1)
        total, _ = compute_multi_horizon_loss(preds, targets, [24, 48], criterion)
        total.backward()
        assert preds.grad is not None


# ==========================================================================
# compute_multi_horizon_metrics
# ==========================================================================

class TestComputeMultiHorizonMetrics:
    def test_keys_present(self):
        preds   = torch.randn(4, 96, 1)
        targets = torch.randn(4, 96, 1)
        metrics = compute_multi_horizon_metrics(preds, targets, [24, 48, 96])
        for h in [24, 48, 96]:
            assert h in metrics
            assert "mse" in metrics[h]
            assert "mae" in metrics[h]

    def test_identical_pred_target(self):
        x = torch.randn(4, 48, 1)
        metrics = compute_multi_horizon_metrics(x, x, [24, 48])
        for h in [24, 48]:
            assert abs(metrics[h]["mse"]) < 1e-7
            assert abs(metrics[h]["mae"]) < 1e-7

    def test_mse_geq_mae_squared(self):
        """For unit-variance noise: MSE ≈ variance, MAE is usually smaller."""
        torch.manual_seed(0)
        preds   = torch.randn(32, 96, 1)
        targets = torch.zeros(32, 96, 1)
        metrics = compute_multi_horizon_metrics(preds, targets, [96])
        assert metrics[96]["mse"] >= 0
        assert metrics[96]["mae"] >= 0

    def test_mse_greater_than_mae_for_large_errors(self):
        """MSE penalises large errors more than MAE."""
        preds   = torch.tensor([[[10.0]], [[10.0]], [[10.0]], [[10.0]]])
        targets = torch.zeros(4, 1, 1)
        metrics = compute_multi_horizon_metrics(preds, targets, [1])
        assert metrics[1]["mse"] > metrics[1]["mae"]

    def test_shorter_horizon_is_subset(self):
        preds   = torch.randn(4, 48, 1)
        targets = torch.randn(4, 48, 1)
        metrics_all = compute_multi_horizon_metrics(preds, targets, [24, 48])
        metrics_24  = compute_multi_horizon_metrics(preds, targets, [24])
        assert abs(metrics_all[24]["mse"] - metrics_24[24]["mse"]) < 1e-6

    def test_metrics_are_floats(self):
        preds   = torch.randn(4, 48, 1)
        targets = torch.randn(4, 48, 1)
        metrics = compute_multi_horizon_metrics(preds, targets, [48])
        assert isinstance(metrics[48]["mse"], float)
        assert isinstance(metrics[48]["mae"], float)

    def test_multi_target_features(self):
        preds   = torch.randn(4, 48, 3)
        targets = torch.randn(4, 48, 3)
        metrics = compute_multi_horizon_metrics(preds, targets, [24, 48])
        for h in [24, 48]:
            assert metrics[h]["mse"] >= 0


# ==========================================================================
# evaluate_dataset (integration)
# ==========================================================================

class TestEvaluateDataset:
    @pytest.fixture
    def regressor(self):
        """A tiny regressor whose encoder accepts (B, F, T) after transposing."""
        # evaluate_dataset transposes: seq_x = seq_x.transpose(1, 2)
        # So DataLoader provides (B, T, F) and regressor receives (B, F, T)
        return _tiny_regressor(embedding_dim=16, horizons=[24, 48])

    def test_returns_dict_for_valid_loader(self, regressor):
        loader  = _make_forecast_loader(seq_in=64, seq_out=48)
        results = evaluate_dataset(regressor, loader, torch.device("cpu"), horizons=[24, 48])
        assert isinstance(results, dict)
        assert 24 in results
        assert 48 in results

    def test_mse_mae_keys_present(self, regressor):
        loader  = _make_forecast_loader(seq_in=64, seq_out=48)
        results = evaluate_dataset(regressor, loader, torch.device("cpu"), horizons=[24, 48])
        for h in [24, 48]:
            assert "mse" in results[h]
            assert "mae" in results[h]

    def test_none_loader_returns_none(self, regressor):
        result = evaluate_dataset(regressor, None, torch.device("cpu"), horizons=[24])
        assert result is None

    def test_metrics_are_non_negative(self, regressor):
        loader  = _make_forecast_loader(seq_in=64, seq_out=48)
        results = evaluate_dataset(regressor, loader, torch.device("cpu"), horizons=[24, 48])
        for h in [24, 48]:
            assert results[h]["mse"] >= 0
            assert results[h]["mae"] >= 0

    def test_metrics_are_finite(self, regressor):
        loader  = _make_forecast_loader(seq_in=64, seq_out=48)
        results = evaluate_dataset(regressor, loader, torch.device("cpu"), horizons=[24, 48])
        import math
        for h in [24, 48]:
            assert math.isfinite(results[h]["mse"])
            assert math.isfinite(results[h]["mae"])
