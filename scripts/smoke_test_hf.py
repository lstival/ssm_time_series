"""Smoke test for CM-Mamba HF models using the new cm_mamba package structure."""

import yaml
from pathlib import Path
import torch
import sys

# Ensure src is in python path
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.append(str(src_path))

from cm_mamba.hf.forecasting import CM_MambaForecastConfig, CM_MambaForecastModel

def load_config_from_yaml(path):
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    print(f"Loaded config from {path}")
    return config_dict

def test_model(config_dict, model_name):
    print(f"\n--- Testing {model_name} ---")
    
    # Extract config parameters
    model_cfg = config_dict.get("model", {})
    forecast_cfg = config_dict.get("forecast", {})
    
    # Create Config object
    config = CM_MambaForecastConfig(
        input_dim=model_cfg.get("input_dim", 32),
        token_size=model_cfg.get("token_size", 32),
        model_dim=model_cfg.get("model_dim", 128),
        embedding_dim=model_cfg.get("embedding_dim", 128),
        depth=model_cfg.get("depth", 8),
        state_dim=model_cfg.get("state_dim", 16),
        conv_kernel=model_cfg.get("conv_kernel", 4),
        expand_factor=model_cfg.get("expand_factor", 1.5),
        dropout=model_cfg.get("dropout", 0.05),
        pooling=model_cfg.get("pooling", "mean"),
        encoder_type=model_cfg.get("encoder_type", "temporal"),
        use_dual_encoder=model_cfg.get("use_dual_encoder", False),
        horizons=forecast_cfg.get("horizons", [96]),
        target_features=forecast_cfg.get("target_features", 1),
        mlp_hidden_dim=forecast_cfg.get("mlp_hidden_dim", 512),
        head_dropout=forecast_cfg.get("head_dropout", 0.1),
        freeze_encoder=forecast_cfg.get("freeze_encoder", True),
    )
    
    # Initialize model (random weights)
    try:
        model = CM_MambaForecastModel(config)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"FAILED to initialize model: {e}")
        return False

    # Create dummy input: (Batch, Time, Features)
    # Time must be at least token_size
    batch_size = 2
    time_len = 384
    features = config.input_dim
    dummy_input = torch.randn(batch_size, time_len, features)
    
    print(f"Running forward pass with input shape: {dummy_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # Verify output shape: (Batch, MaxHorizon, TargetFeatures)
        max_horizon = max(config.horizons)
        expected_shape = (batch_size, max_horizon, config.target_features)
        
        if output.shape == expected_shape:
            print("Output shape verification: PASSED")
            return True
        else:
            print(f"Output shape verification: FAILED. Expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"FAILED during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    root_dir = Path(__file__).resolve().parents[1]
    configs_dir = root_dir / "src" / "cm_mamba" / "configs" / "hf"
    
    metrics = {}
    
    # Test Mini
    mini_config_path = configs_dir / "cm_mamba_mini.yaml"
    if mini_config_path.exists():
        mini_cfg = load_config_from_yaml(mini_config_path)
        metrics["mini"] = test_model(mini_cfg, "cm-mamba-mini")
    else:
        print(f"Config not found: {mini_config_path}")
        metrics["mini"] = "skipped"

    # Test Tiny
    tiny_config_path = configs_dir / "cm_mamba_tiny.yaml"
    if tiny_config_path.exists():
        tiny_cfg = load_config_from_yaml(tiny_config_path)
        metrics["tiny"] = test_model(tiny_cfg, "cm-mamba-tiny")
    else:
        print(f"Config not found: {tiny_config_path}")
        metrics["tiny"] = "skipped"

    print("\n--- Summary ---")
    for k, v in metrics.items():
        print(f"{k}: {'PASSED' if v is True else 'FAILED' if v is False else v}")

if __name__ == "__main__":
    main()
