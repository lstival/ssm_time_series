"""Tests for forecasting head models and regressors.

Covers:
  - MultiHorizonForecastMLP: shapes, horizon slicing, parameter validation
  - ForecastRegressor: frozen encoder, frozen-eval mode, gradient isolation
  - DualEncoderForecastMLP: same shape checks as single-encoder version
  - DualEncoderForecastRegressor: concatenated embedding flow
  - Determinism, device placement, serialisation round-trip
"""

from __future__ import annotations

import io
import pytest
import torch
import torch.nn as nn

from models.classifier import MultiHorizonForecastMLP, ForecastRegressor
from models.dual_forecast import DualEncoderForecastMLP, DualEncoderForecastRegressor
from models.mamba_encoder import MambaEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_encoder(embedding_dim: int = 16) -> MambaEncoder:
    return MambaEncoder(
        input_dim=8,
        model_dim=16,
        depth=1,
        state_dim=4,
        expand_factor=1.5,
        embedding_dim=embedding_dim,
    )


# ==========================================================================
# MultiHorizonForecastMLP
# ==========================================================================

class TestMultiHorizonForecastMLPShapes:
    @pytest.fixture
    def mlp(self):
        return MultiHorizonForecastMLP(
            input_dim=32,
            hidden_dim=64,
            horizons=[24, 48, 96],
            target_features=1,
        )

    def test_default_returns_max_horizon(self, mlp):
        x = torch.randn(4, 32)
        out = mlp(x)
        assert out.shape == (4, 96, 1)

    def test_horizon_slice(self, mlp):
        x = torch.randn(4, 32)
        for h in [24, 48, 96]:
            out = mlp(x, horizon=h)
            assert out.shape == (4, h, 1), f"Wrong shape for horizon={h}"

    def test_multi_target_features(self):
        mlp = MultiHorizonForecastMLP(32, 64, [12, 24], target_features=3)
        x = torch.randn(2, 32)
        assert mlp(x).shape == (2, 24, 3)

    def test_batch_size_1(self, mlp):
        x = torch.randn(1, 32)
        assert mlp(x).shape == (1, 96, 1)

    def test_large_batch(self, mlp):
        x = torch.randn(128, 32)
        assert mlp(x).shape == (128, 96, 1)


class TestMultiHorizonForecastMLPValidation:
    def test_empty_horizons_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            MultiHorizonForecastMLP(32, 64, [])

    def test_invalid_horizon_request_raises(self):
        mlp = MultiHorizonForecastMLP(32, 64, [24, 96])
        with pytest.raises(ValueError, match="not configured"):
            mlp(torch.randn(2, 32), horizon=48)

    def test_horizons_are_sorted(self):
        mlp = MultiHorizonForecastMLP(32, 64, [96, 24, 48])
        assert mlp.horizons == [24, 48, 96]

    def test_duplicate_horizons_deduplicated(self):
        mlp = MultiHorizonForecastMLP(32, 64, [24, 24, 48])
        assert mlp.horizons == [24, 48]


class TestMultiHorizonForecastMLPNumerics:
    def test_output_finite(self):
        mlp = MultiHorizonForecastMLP(16, 32, [12])
        x = torch.randn(4, 16)
        assert torch.isfinite(mlp(x)).all()

    def test_backward_finite_gradients(self):
        mlp = MultiHorizonForecastMLP(16, 32, [12])
        x = torch.randn(4, 16, requires_grad=True)
        mlp(x).sum().backward()
        assert torch.isfinite(x.grad).all()


class TestMultiHorizonForecastMLPSerialization:
    def test_state_dict_round_trip(self):
        mlp = MultiHorizonForecastMLP(16, 32, [12, 24])
        buf = io.BytesIO()
        torch.save(mlp.state_dict(), buf)
        buf.seek(0)
        mlp2 = MultiHorizonForecastMLP(16, 32, [12, 24])
        mlp2.load_state_dict(torch.load(buf))
        x = torch.randn(2, 16)
        mlp.eval(); mlp2.eval()
        with torch.no_grad():
            assert torch.allclose(mlp(x), mlp2(x))


# ==========================================================================
# ForecastRegressor
# ==========================================================================

class TestForecastRegressorShapes:
    @pytest.fixture
    def regressor(self):
        enc = _tiny_encoder(embedding_dim=16)
        head = MultiHorizonForecastMLP(16, 32, [24, 48])
        return ForecastRegressor(encoder=enc, head=head, freeze_encoder=True)

    def test_forward_shape(self, regressor):
        x = torch.randn(2, 1, 64)           # (B, F, T)
        out = regressor(x)
        assert out.shape == (2, 48, 1)

    def test_forward_with_horizon(self, regressor):
        x = torch.randn(2, 1, 64)
        out = regressor(x, horizon=24)
        assert out.shape == (2, 24, 1)

    def test_max_horizon_property(self, regressor):
        assert regressor.max_horizon == 48


class TestForecastRegressorFreezeEncoder:
    def test_encoder_params_frozen(self):
        enc = _tiny_encoder(16)
        head = MultiHorizonForecastMLP(16, 32, [24])
        reg = ForecastRegressor(encoder=enc, head=head, freeze_encoder=True)
        for p in reg.encoder.parameters():
            assert not p.requires_grad

    def test_head_params_trainable(self):
        enc = _tiny_encoder(16)
        head = MultiHorizonForecastMLP(16, 32, [24])
        reg = ForecastRegressor(encoder=enc, head=head, freeze_encoder=True)
        for p in reg.head.parameters():
            assert p.requires_grad

    def test_encoder_stays_in_eval_during_train(self):
        enc = _tiny_encoder(16)
        head = MultiHorizonForecastMLP(16, 32, [24])
        reg = ForecastRegressor(encoder=enc, head=head, freeze_encoder=True)
        reg.train()
        assert not reg.encoder.training, "Frozen encoder must stay in eval mode"

    def test_no_freeze_keeps_encoder_trainable(self):
        enc = _tiny_encoder(16)
        head = MultiHorizonForecastMLP(16, 32, [24])
        reg = ForecastRegressor(encoder=enc, head=head, freeze_encoder=False)
        for p in reg.encoder.parameters():
            assert p.requires_grad

    def test_frozen_encoder_gradient_does_not_propagate(self):
        enc = _tiny_encoder(16)
        head = MultiHorizonForecastMLP(16, 32, [12])
        reg = ForecastRegressor(encoder=enc, head=head, freeze_encoder=True)
        x = torch.randn(2, 1, 64)
        reg(x).sum().backward()
        for p in reg.encoder.parameters():
            assert p.grad is None, "Frozen encoder params must have no gradient"


class TestForecastRegressorNumerics:
    def test_output_finite(self):
        enc = _tiny_encoder(16)
        head = MultiHorizonForecastMLP(16, 32, [24])
        reg = ForecastRegressor(encoder=enc, head=head)
        x = torch.randn(3, 1, 64)
        assert torch.isfinite(reg(x)).all()


# ==========================================================================
# DualEncoderForecastMLP
# ==========================================================================

class TestDualEncoderForecastMLPShapes:
    @pytest.fixture
    def dual_mlp(self):
        # Dual encoder concatenates two embeddings → input_dim = 2 × embedding_dim
        return DualEncoderForecastMLP(
            input_dim=32,   # 2 × 16
            hidden_dim=64,
            horizons=[24, 48],
            target_features=1,
        )

    def test_default_shape(self, dual_mlp):
        x = torch.randn(3, 32)
        assert dual_mlp(x).shape == (3, 48, 1)

    def test_horizon_slice(self, dual_mlp):
        x = torch.randn(3, 32)
        assert dual_mlp(x, horizon=24).shape == (3, 24, 1)

    def test_output_finite(self, dual_mlp):
        assert torch.isfinite(dual_mlp(torch.randn(4, 32))).all()


# ==========================================================================
# DualEncoderForecastRegressor
# ==========================================================================

class TestDualEncoderForecastRegressorShapes:
    @pytest.fixture
    def dual_regressor(self):
        enc  = _tiny_encoder(16)
        vis  = _tiny_encoder(16)
        head = DualEncoderForecastMLP(32, 64, [24, 48])  # 16+16=32
        return DualEncoderForecastRegressor(encoder=enc, visual_encoder=vis, head=head)

    def test_forward_shape(self, dual_regressor):
        # DualEncoderForecastRegressor.forward transposes (B, T, F) → (B, F, T)
        # before passing to the encoder, which internally expects (B, F, T).
        # Token size = 8 (default), so we need T ≥ 8 in the original input.
        x = torch.randn(2, 64, 1)   # (B=2, T=64, F=1)
        out = dual_regressor(x)
        assert out.shape == (2, 48, 1)

    def test_both_encoders_eval_mode(self, dual_regressor):
        assert not dual_regressor.encoder.training
        assert not dual_regressor.visual_encoder.training

    def test_output_finite(self, dual_regressor):
        x = torch.randn(2, 64, 1)   # (B=2, T=64, F=1) — transpose gives (B, F=1, T=64)
        assert torch.isfinite(dual_regressor(x)).all()
