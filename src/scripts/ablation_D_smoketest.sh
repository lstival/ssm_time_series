#!/bin/bash
#SBATCH --comment=ablation_D_smoketest
#SBATCH --time=60
#SBATCH --mem=20000
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/ablation_D_smoketest/ablation_D_smoketest_%j.out
#SBATCH --error=logs/ablation_D_smoketest/ablation_D_smoketest_%j.err
#SBATCH --job-name=ablation_D_smoke
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_D_smoketest

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# Quick smoke test of fixed Ablation D
# Uses minimal epochs and a single dataset to validate the fix
# Expected duration: ~5 minutes (vs 68 minutes for full ablation D)

echo "Starting Ablation D smoke test..."
echo "Testing with minimal configuration:"
echo "  - Train epochs: 2 (vs 20)"
echo "  - Probe epochs: 3 (vs 30)"
echo "  - Single dataset: ETTm1.csv (fast)"
echo ""

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/ablation_D_visual_repr.py \
    --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_clip.yaml \
    --train_epochs 2 \
    --probe_epochs 3

EXIT_CODE=$?

echo ""
echo "Smoke test completed with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Ablation D fix validated"
else
    echo "❌ FAILURE: Ablation D still has issues"
fi

exit $EXIT_CODE

# Job: Quick validation that Ablation D CUDA error is fixed
#
# Previous error:
#   RuntimeError: CUDA error: device-side assert triggered
#   Caused by NaN/inf values in gradient computation
#
# Fix implemented:
#   1. Tensor shape validation
#   2. Numerical stability (clamping, clipping)
#   3. NaN/inf handling before backward pass
#   4. Gradient clipping to prevent overflow
#
# This smoke test runs with minimal configuration to quickly verify:
#   - Encoder loads without error
#   - Data loading works
#   - Training loop completes without CUDA error
#   - Probe training succeeds
#
# If successful, full ablation can be rerun
