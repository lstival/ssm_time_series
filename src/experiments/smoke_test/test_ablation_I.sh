#!/bin/bash
# Smoke test for Ablation I — runs 1 method with minimal data to catch errors quickly

set -e

REPO_ROOT=$(pwd)
RESULTS_DIR="${REPO_ROOT}/results/ablation_I_smoke_test"
CONFIG="${REPO_ROOT}/src/configs/lotsa_clip.yaml"

mkdir -p "$RESULTS_DIR"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "Smoke Test: Ablation I — Advanced Multivariate RP Methods"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Testing 1 method (channel_stacking) to validate script integrity"
echo "Repository: $REPO_ROOT"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════════"

# Run with just one method, 1 epoch training, 1 epoch probing
python "${REPO_ROOT}/src/experiments/ablation_I_multivariate_rp.py" \
    --config "$CONFIG" \
    --train_epochs 1 \
    --probe_epochs 1 \
    --results_dir "$RESULTS_DIR" \
    --methods channel_stacking \
    --device cuda \
    --seed 42

echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ Smoke test PASSED"
echo "Results saved to: $RESULTS_DIR/"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════════"
