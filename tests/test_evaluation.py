"""Tests for evaluation utilities."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ssm_time_series.tasks.forecast_utils import discover_icml_datasets, load_trained_model
from ssm_time_series.utils.nn import default_device


def test_model_loading() -> None:
    """Test loading the trained model from an env-provided checkpoint."""
    checkpoint_env = os.getenv("SSM_EVAL_CHECKPOINT")
    if not checkpoint_env:
        pytest.skip("SSM_EVAL_CHECKPOINT not set")

    checkpoint_path = Path(checkpoint_env).expanduser().resolve()
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    device = default_device()
    model, checkpoint_info = load_trained_model(checkpoint_path, device)
    assert model is not None
    assert "horizons" in checkpoint_info


def test_dataset_discovery() -> None:
    """Test discovering ICML datasets from an env-provided directory."""
    icml_env = os.getenv("SSM_ICML_DATA_DIR")
    if not icml_env:
        pytest.skip("SSM_ICML_DATA_DIR not set")

    icml_dir = Path(icml_env).expanduser().resolve()
    if not icml_dir.exists():
        pytest.skip(f"ICML dataset dir not found: {icml_dir}")

    datasets = discover_icml_datasets(icml_dir)
    assert isinstance(datasets, list)