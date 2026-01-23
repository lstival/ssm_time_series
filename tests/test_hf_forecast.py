"""Tests for CM-Mamba forecasting HF interface."""

from __future__ import annotations

from pathlib import Path

import torch

from ssm_time_series.hf.forecasting import CM_MambaForecastConfig, CM_MambaForecastModel


def _build_dual_config() -> CM_MambaForecastConfig:
    return CM_MambaForecastConfig(
        input_dim=4,
        token_size=4,
        model_dim=16,
        embedding_dim=8,
        depth=1,
        state_dim=4,
        conv_kernel=3,
        expand_factor=1.5,
        dropout=0.0,
        pooling="mean",
        encoder_type="temporal",
        use_dual_encoder=True,
        horizons=[2, 4],
        target_features=1,
        mlp_hidden_dim=8,
        head_dropout=0.0,
        freeze_encoder=False,
    )


def test_forecast_from_checkpoint(tmp_path: Path) -> None:
    config = _build_dual_config()
    model = CM_MambaForecastModel(config)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    loaded = CM_MambaForecastModel.from_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    x = torch.randn(2, 8, config.input_dim)
    y = loaded(x)
    assert y.shape == (2, max(config.horizons), config.target_features)
    assert not torch.isnan(y).any()


def test_encoder_only_loading(tmp_path: Path) -> None:
    config = _build_dual_config()
    model = CM_MambaForecastModel(config)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    encoders = CM_MambaForecastModel.from_checkpoint_encoder_only(
        config=config,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    assert isinstance(encoders, tuple)
    assert len(encoders) == 2
