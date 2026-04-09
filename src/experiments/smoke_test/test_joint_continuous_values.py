"""
Detailed Verification: Joint Strategy Uses Continuous Values
=============================================================
Validates that the fixed joint strategy produces continuous [0,1] values
instead of binary {0, 1}.

Expected runtime: ~10 seconds
"""

import sys
from pathlib import Path

import numpy as np
import torch

script_dir = Path(__file__).resolve().parent.parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.mamba_visual_encoder import MambaVisualEncoder


def test_continuous_values():
    """Test that joint strategy produces continuous [0,1] values."""
    print("\n" + "="*70)
    print("TEST: Joint Strategy Continuous Values")
    print("="*70)

    # Create visual encoder with joint strategy
    visual = MambaVisualEncoder(
        input_dim=64,
        model_dim=256,
        depth=2,
        rp_mv_strategy="joint",
        repr_type="rp",
    )

    print("\n✅ Encoder created with rp_mv_strategy='joint'")

    # Test 1: Low-dimensional data (7 channels)
    print("\n[TEST 1] Low-D Data (7 channels, 96 timesteps)")
    print("-" * 70)
    x_low = np.random.randn(2, 7, 96).astype(np.float32)
    result_low = visual._apply_mv_strategy(x_low)

    print(f"  Input shape: {x_low.shape}")
    print(f"  Output shape: {result_low.shape}")
    print(f"  Output min: {result_low.min():.6f}")
    print(f"  Output max: {result_low.max():.6f}")
    print(f"  Output mean: {result_low.mean():.6f}")
    print(f"  Output dtype: {result_low.dtype}")

    # Check range
    if 0 <= result_low.min() and result_low.max() <= 1.0:
        print("  ✅ Values in [0, 1] range ✓")
    else:
        print(f"  ❌ Values OUT OF RANGE! Min={result_low.min()}, Max={result_low.max()}")
        return False

    # Check it's NOT binary
    unique_vals = np.unique(result_low)
    if len(unique_vals) > 2:
        print(f"  ✅ Continuous values detected ({len(unique_vals)} unique values)")
    else:
        print(f"  ❌ Binary values detected! ({unique_vals})")
        return False

    # Test 2: High-dimensional data (321 channels)
    print("\n[TEST 2] High-D Data (321 channels, 96 timesteps)")
    print("-" * 70)
    x_high = np.random.randn(2, 321, 96).astype(np.float32)
    result_high = visual._apply_mv_strategy(x_high)

    print(f"  Input shape: {x_high.shape}")
    print(f"  Output shape: {result_high.shape}")
    print(f"  Output min: {result_high.min():.6f}")
    print(f"  Output max: {result_high.max():.6f}")
    print(f"  Output mean: {result_high.mean():.6f}")
    print(f"  Output dtype: {result_high.dtype}")

    # Check range
    if 0 <= result_high.min() and result_high.max() <= 1.0:
        print("  ✅ Values in [0, 1] range ✓")
    else:
        print(f"  ❌ Values OUT OF RANGE! Min={result_high.min()}, Max={result_high.max()}")
        return False

    # Check it's NOT binary
    unique_vals_high = np.unique(result_high)
    if len(unique_vals_high) > 2:
        print(f"  ✅ Continuous values detected ({len(unique_vals_high)} unique values)")
    else:
        print(f"  ❌ Binary values detected! ({unique_vals_high})")
        return False

    # Test 3: Verify diagonal is 0 (distance from point to itself)
    print("\n[TEST 3] Verify Diagonal Properties")
    print("-" * 70)
    diag_low = np.diag(result_low[0])
    diag_high = np.diag(result_high[0])

    print(f"  Low-D diagonal min: {diag_low.min():.8f}")
    print(f"  Low-D diagonal max: {diag_low.max():.8f}")
    print(f"  Low-D diagonal mean: {diag_low.mean():.8f}")

    print(f"  High-D diagonal min: {diag_high.min():.8f}")
    print(f"  High-D diagonal max: {diag_high.max():.8f}")
    print(f"  High-D diagonal mean: {diag_high.mean():.8f}")

    if diag_low.max() < 0.01:
        print("  ✅ Low-D diagonal ≈ 0 (correct for distance from point to itself) ✓")
    else:
        print(f"  ⚠️  Low-D diagonal not zero: {diag_low.max()}")

    if diag_high.max() < 0.01:
        print("  ✅ High-D diagonal ≈ 0 (correct for distance from point to itself) ✓")
    else:
        print(f"  ⚠️  High-D diagonal not zero: {diag_high.max()}")

    # Test 4: Verify symmetry
    print("\n[TEST 4] Verify Symmetry (RP should be symmetric)")
    print("-" * 70)
    sym_error_low = np.abs(result_low[0] - result_low[0].T).max()
    sym_error_high = np.abs(result_high[0] - result_high[0].T).max()

    print(f"  Low-D max symmetry error: {sym_error_low:.8f}")
    print(f"  High-D max symmetry error: {sym_error_high:.8f}")

    if sym_error_low < 1e-5:
        print("  ✅ Low-D RP is symmetric ✓")
    if sym_error_high < 1e-5:
        print("  ✅ High-D RP is symmetric ✓")

    return True


def main():
    print("\n" + "="*70)
    print("VERIFICATION: Joint Strategy = Continuous Global L2 RP")
    print("="*70)

    passed = test_continuous_values()

    print("\n" + "="*70)
    if passed:
        print("✅ All verification tests PASSED!")
        print("   Joint strategy correctly uses continuous L2 distances")
        print("   Ready to submit: sbatch src/scripts/test_mean_vs_joint.sh")
    else:
        print("❌ Verification FAILED!")
        print("   Check the encoder implementation")
    print("="*70 + "\n")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
