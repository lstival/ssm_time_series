"""
Smoke Test: Mean vs Joint RP Methods
=====================================
Quick validation that both strategies are properly defined and can be
instantiated without errors.

This tests:
1. Both strategies can be instantiated in MambaVisualEncoder
2. _apply_mv_strategy method works for both
3. No syntax or import errors

Expected runtime: ~5 seconds
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


def test_strategy_instantiation(strategy: str):
    """Test that a strategy can be instantiated without errors."""
    print(f"\n  Testing instantiation: {strategy}")

    try:
        visual = MambaVisualEncoder(
            input_dim=64,
            model_dim=256,
            depth=2,
            rp_mv_strategy=strategy,
            repr_type="rp",
        )
        print(f"    ✅ Encoder created for strategy='{strategy}'")
        print(f"       - rp_mv_strategy: {visual.rp_mv_strategy}")
        return True
    except Exception as e:
        print(f"    ❌ Failed to create encoder: {e}")
        return False


def test_mv_strategy_logic(strategy: str, n_channels: int):
    """Test the _apply_mv_strategy method directly."""
    print(f"\n  Testing {strategy} strategy logic (n_channels={n_channels})")

    try:
        visual = MambaVisualEncoder(
            input_dim=64,
            model_dim=256,
            depth=2,
            rp_mv_strategy=strategy,
            repr_type="rp",
        )

        # Create synthetic data
        x_np = np.random.randn(1, n_channels, 96).astype(np.float32)

        # Apply strategy
        result = visual._apply_mv_strategy(x_np)

        print(f"    ✅ Strategy logic executed successfully")
        print(f"       - Input shape:  {x_np.shape}")
        print(f"       - Output shape: {result.shape}")

        # Validate output
        if strategy == "mean":
            # Should output (1, 1, 96) before RP
            if result.shape == (1, 1, 96):
                print(f"       - Shape valid for 'mean': averaged to 1 channel ✓")
                return True
            else:
                print(f"       - Shape unexpected: {result.shape}")
                return False

        elif strategy == "joint":
            # Should output (1, 96, 96) — L2 distance matrix
            if result.shape == (1, 96, 96):
                print(f"       - Shape valid for 'joint': (1, L, L) RP ✓")
                return True
            else:
                print(f"       - Shape unexpected: {result.shape}")
                return False

    except Exception as e:
        print(f"    ❌ Strategy logic failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("SMOKE TEST: Mean vs Joint RP Strategies")
    print("="*70)

    strategies = ["mean", "joint"]
    test_configs = [
        ("ETTm1-like", 7),
        ("Weather-like", 21),
        ("Traffic-like", 321),
    ]

    all_passed = True

    # Test 1: Instantiation
    print("\n[TEST 1] Instantiation")
    print("-" * 70)
    for strat in strategies:
        passed = test_strategy_instantiation(strat)
        all_passed = all_passed and passed

    # Test 2: Strategy logic
    print("\n[TEST 2] Strategy Logic")
    print("-" * 70)
    for ds_name, n_ch in test_configs:
        print(f"\n  {ds_name} ({n_ch} channels):")
        for strat in strategies:
            passed = test_mv_strategy_logic(strat, n_ch)
            all_passed = all_passed and passed

    print("\n" + "="*70)
    if all_passed:
        print("✅ All smoke tests PASSED!")
        print("   Safe to submit full SLURM job: sbatch src/scripts/test_mean_vs_joint.sh")
    else:
        print("❌ Some tests FAILED. Check errors above.")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
