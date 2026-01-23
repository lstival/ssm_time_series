"""Tests for CM-Mamba HF encoder interfaces."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ssm_time_series.hf.cm_mamba import CM_MambaCombined, CM_MambaTemporal, CM_MambaVisual
from ssm_time_series.models.mamba_encoder import MambaEncoder


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with a mock model."""
    config = {
        "input_dim": 32,
        "token_size": 16,
        "model_dim": 64,
        "embedding_dim": 128,
        "depth": 2,
        "state_dim": 8,
        "pooling": "mean",
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    encoder = MambaEncoder(
        input_dim=config["input_dim"],
        token_size=config["token_size"],
        model_dim=config["model_dim"],
        embedding_dim=config["embedding_dim"],
        depth=config["depth"],
        state_dim=config["state_dim"],
    )
    state_dict = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}
    torch.save(state_dict, tmp_path / "pytorch_model.bin")

    return tmp_path


def test_temporal_loading(temp_model_dir: Path) -> None:
    """Test loading a temporal encoder from a local path."""
    model = CM_MambaTemporal.from_pretrained(temp_model_dir, device="cpu")
    assert isinstance(model, CM_MambaTemporal)
    assert model.config.embedding_dim == 128

    x = torch.randn(2, 96, 32)
    out = model(x)
    assert out.shape == (2, 128)


def test_visual_loading(temp_model_dir: Path) -> None:
    """Test loading a visual encoder from a local path."""
    config_path = temp_model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_dim"] = 3
    config["pooling"] = "cls"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    model = CM_MambaVisual.from_pretrained(temp_model_dir, device="cpu")
    assert isinstance(model, CM_MambaVisual)

    x = torch.randn(2, 96, 3)
    out = model(x)
    assert out.shape == (2, 128)


def test_combined_loading(temp_model_dir: Path) -> None:
    """Test loading a combined encoder."""
    model = CM_MambaCombined.from_pretrained(
        temporal_repo_or_path=temp_model_dir,
        visual_repo_or_path=temp_model_dir,
        device="cpu",
    )
    assert isinstance(model, CM_MambaCombined)
    assert model.embedding_dim == 256

    x = torch.randn(2, 96, 32)
    out = model(x)
    assert out.shape == (2, 256)
