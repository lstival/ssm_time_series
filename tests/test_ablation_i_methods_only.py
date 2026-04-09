#!/usr/bin/env python3
"""
Quick test of Ablation I RP methods — no training, just validate implementations
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import the ablation script
from experiments.ablation_I_multivariate_rp import (
    rp_channel_stacking,
    rp_global_l2,
    rp_jrp_hadamard,
    rp_crp_block,
    rp_ms_fusion_concat,
)

def test_methods():
    """Test each RP method with synthetic data"""
    print("=" * 80)
    print("Ablation I RP Methods — Validation Test")
    print("=" * 80)

    # Create synthetic multivariate time series
    n_samples = 2
    n_channels = 5
    length = 32

    x = np.random.randn(n_samples, n_channels, length).astype(np.float32)
    print(f"\nTest input shape: {x.shape}")
    print(f"  Samples: {n_samples}, Channels: {n_channels}, Length: {length}\n")

    methods = {
        "Channel Stacking": (rp_channel_stacking, "Stack per-channel RPs"),
        "Global L2": (rp_global_l2, "Single RP from L2 distance"),
        "JRP (Hadamard)": (rp_jrp_hadamard, "Hadamard product of per-channel RPs"),
        "CRP (Block)": (rp_crp_block, "Block matrix with cross-RPs"),
        "Multi-Scale Fusion": (rp_ms_fusion_concat, "Multiple scales concatenated"),
    }

    results = {}

    for name, (func, desc) in methods.items():
        try:
            print(f"Testing: {name}")
            print(f"  Description: {desc}")

            # Run method
            output = func(x)

            # Check output
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Min/Max: {output.min():.4f} / {output.max():.4f}")
            print(f"  NaN count: {np.isnan(output).sum()}")

            # Validate
            assert output.dtype == np.float32, f"Expected float32, got {output.dtype}"
            assert not np.isnan(output).any(), "Output contains NaN values"
            assert output.min() >= -0.1, f"Output has invalid min: {output.min()}"
            assert output.max() <= 1.1, f"Output has invalid max: {output.max()}"

            results[name] = "✅ PASS"
            print(f"  Status: ✅ PASS\n")

        except Exception as e:
            results[name] = f"❌ FAIL: {str(e)}"
            print(f"  Status: ❌ FAIL: {e}\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, status in results.items():
        print(f"{name:25s} {status}")

    passed = sum(1 for s in results.values() if s.startswith("✅"))
    total = len(results)
    print(f"\nTotal: {passed}/{total} methods passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = test_methods()
    sys.exit(exit_code)
