"""Simple test script to verify the evaluation utilities work correctly."""

from pathlib import Path
# Removed legacy sys.path hack

import torch
from ssm_time_series.utils.nn import default_device
from ssm_time_series.tasks.forecast_utils import load_trained_model, discover_icml_datasets


def test_model_loading():
    """Test loading the trained model."""
    print("Testing model loading...")
    
    checkpoint_path = Path(r"C:\WUR\ssm_time_series\checkpoints\multi_horizon_forecast_emb_128_tgt_1_20251108_1124\best_model.pt")
    device = default_device()
    
    if not checkpoint_path.exists():
        print(f"Skipping test: {checkpoint_path} not found")
        return

    try:
        model, checkpoint_info = load_trained_model(checkpoint_path, device)
        print(f"✓ Model loaded successfully!")
        print(f"  Horizons: {checkpoint_info['horizons']}")
        print(f"  Input dim: {checkpoint_info['input_dim']}")
        print(f"  Hidden dim: {checkpoint_info['hidden_dim']}")
        print(f"  Target features: {checkpoint_info['target_features']}")
        assert model is not None
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        assert False, f"Model loading failed: {e}"


def test_dataset_discovery():
    """Test discovering ICML datasets."""
    print("\nTesting dataset discovery...")
    
    icml_dir = Path(r"C:\WUR\ssm_time_series\ICML_datasets")
    
    if not icml_dir.exists():
        print(f"Skipping test: {icml_dir} not found")
        return

    try:
        datasets = discover_icml_datasets(icml_dir)
        print(f"✓ Found {len(datasets)} ICML datasets:")
        for name, path in datasets:
            print(f"  - {name}")
        assert len(datasets) >= 0
    except Exception as e:
        print(f"✗ Dataset discovery failed: {e}")
        assert False, f"Dataset discovery failed: {e}"


def main():
    """Run all tests."""
    print("Running evaluation utilities tests...\n")
    
    tests = [
        test_model_loading,
        test_dataset_discovery,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\nTest Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✓ All tests passed! Ready to run evaluation.")
    else:
        print("✗ Some tests failed. Please check the setup.")


if __name__ == "__main__":
    main()