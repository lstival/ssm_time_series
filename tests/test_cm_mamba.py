"""Tests for CM_Mamba interface."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import pytest
from ssm_time_series.hf.cm_mamba import CM_MambaTemporal, CM_MambaVisual, CM_MambaCombined


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory with a mock model."""
    tmpdir = tempfile.mkdtemp()
    model_dir = Path(tmpdir)
    
    # Mock config
    config = {
        "input_dim": 32,
        "model_dim": 64,
        "embedding_dim": 128,
        "depth": 2,
        "state_dim": 8,
        "pooling": "mean"
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
        
    # Mock weights
    # We need to simulate the structure expected by load_state_dict
    # Since CM_MambaTemporal wraps MambaEncoder as self.encoder
    from ssm_time_series.models.mamba_encoder import MambaEncoder
    encoder = MambaEncoder(
        input_dim=config["input_dim"],
        model_dim=config["model_dim"],
        embedding_dim=config["embedding_dim"],
        depth=config["depth"],
        state_dim=config["state_dim"]
    )
    torch.save({"model_state_dict": encoder.state_dict()}, model_dir / "pytorch_model.pt")
    
    yield model_dir
    shutil.rmtree(tmpdir)


def test_temporal_loading(temp_model_dir):
    """Test loading a temporal encoder from a local path."""
    model = CM_MambaTemporal.from_pretrained(temp_model_dir, device="cpu")
    assert isinstance(model, CM_MambaTemporal)
    assert model.embedding_dim == 128
    
    # Test inference
    x = torch.randn(2, 32, 96) # (batch, features, seq)
    out = model(x)
    assert out.shape == (2, 128)


def test_visual_loading(temp_model_dir):
    """Test loading a visual encoder from a local path."""
    # Update config for visual
    config_path = temp_model_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config["input_dim"] = 3 # channels
    config["pooling"] = "cls"
    with open(config_path, "w") as f:
        json.dump(config, f)
        
    model = CM_MambaVisual.from_pretrained(temp_model_dir, device="cpu")
    assert isinstance(model, CM_MambaVisual)
    
    # Test inference
    x = torch.randn(2, 3, 96) # (batch, channels, seq)
    out = model(x)
    assert out.shape == (2, 128)


def test_combined_loading(temp_model_dir):
    """Test loading a combined encoder."""
    # We'll use the same dir for both for simplicity in the test
    model = CM_MambaCombined.from_pretrained(
        temporal_repo_or_path=temp_model_dir,
        visual_repo_or_path=temp_model_dir,
        device="cpu"
    )
    assert isinstance(model, CM_MambaCombined)
    assert model.embedding_dim == 256 # 128 + 128
    
    # Test inference
    x = torch.randn(2, 32, 96) # Using temporal-like input (will be shared)
    # Note: CM_MambaCombined passes same x to both. 
    # In practice inputs might differ but the interface supports it.
    out = model(x)
    assert out.shape == (2, 256)

if __name__ == "__main__":
    pytest.main([__file__])
