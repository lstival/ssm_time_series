#!/bin/bash
# Smoke tests for ablations A-F using ICML datasets (local CSVs).
# Purpose: validate code correctness and directional behaviour before
#          running the full LOTSA-scale ablations.
#
# Runtime: ~5-15 min on GPU, ~30-60 min on CPU
#
# Usage (from repo root):
#   bash src/experiments/smoke_test/run_smoke_tests.sh
#
# To run a single ablation:
#   SMOKE_ABLATIONS="A B" bash src/experiments/smoke_test/run_smoke_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${SRC_DIR}/.." && pwd)"

SMOKE_CFG="${SCRIPT_DIR}/smoke_config.yaml"
ICML_DIR="${REPO_DIR}/ICML_datasets"
ETT_DIR="${ICML_DIR}/ETT-small"
RESULTS_DIR="${REPO_DIR}/results/smoke"
CKPT_DIR="${REPO_DIR}/checkpoints/smoke"

# Which ablations to run (default: all)
SMOKE_ABLATIONS="${SMOKE_ABLATIONS:-A B C D E F}"

TRAIN_EPOCHS=3
PROBE_EPOCHS=5

echo "========================================================"
echo "  CM-Mamba Ablation Smoke Tests"
echo "  config : ${SMOKE_CFG}"
echo "  data   : ${ETT_DIR}"
echo "  results: ${RESULTS_DIR}"
echo "========================================================"

mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}"

cd "${SRC_DIR}"

pass=0
fail=0

run_ablation() {
    local name="$1"; shift
    echo ""
    echo "-------- Ablation ${name} --------"
    if python3 "$@" 2>&1; then
        echo "[PASS] Ablation ${name}"
        pass=$((pass + 1))
    else
        echo "[FAIL] Ablation ${name}"
        fail=$((fail + 1))
    fi
}

for abl in ${SMOKE_ABLATIONS}; do
    case "${abl}" in

    A)
        run_ablation A experiments/ablation_A_mv_rp.py \
            --config "${SMOKE_CFG}" \
            --train_epochs ${TRAIN_EPOCHS} \
            --probe_epochs ${PROBE_EPOCHS} \
            --pretrain_data_dir "${ETT_DIR}" \
            --data_dir "${ICML_DIR}/ETT-small" \
            --strategies per_channel mean \
            --results_dir "${RESULTS_DIR}/ablation_A"
        ;;

    B)
        run_ablation B experiments/ablation_B_encoder_arch.py \
            --config "${SMOKE_CFG}" \
            --train_epochs ${TRAIN_EPOCHS} \
            --probe_epochs ${PROBE_EPOCHS} \
            --pretrain_data_dir "${ETT_DIR}" \
            --data_dir "${ICML_DIR}/ETT-small" \
            --variants no_visual sep_mamba_1d \
            --results_dir "${RESULTS_DIR}/ablation_B"
        ;;

    C)
        run_ablation C experiments/ablation_C_alignment.py \
            --config "${SMOKE_CFG}" \
            --train_epochs ${TRAIN_EPOCHS} \
            --probe_epochs ${PROBE_EPOCHS} \
            --pretrain_data_dir "${ETT_DIR}" \
            --data_dir "${ICML_DIR}/ETT-small" \
            --variants clip_symm unimodal_temporal \
            --results_dir "${RESULTS_DIR}/ablation_C"
        ;;

    D)
        run_ablation D experiments/ablation_D_visual_repr.py \
            --config "${SMOKE_CFG}" \
            --train_epochs ${TRAIN_EPOCHS} \
            --probe_epochs ${PROBE_EPOCHS} \
            --pretrain_data_dir "${ETT_DIR}" \
            --data_dir "${ICML_DIR}/ETT-small" \
            --repr_types rp gasf \
            --results_dir "${RESULTS_DIR}/ablation_D"
        ;;

    E)
        run_ablation E experiments/ablation_E_patch_length.py \
            --config "${SMOKE_CFG}" \
            --train_epochs ${TRAIN_EPOCHS} \
            --probe_epochs ${PROBE_EPOCHS} \
            --pretrain_data_dir "${ETT_DIR}" \
            --data_dir "${ICML_DIR}/ETT-small" \
            --patch_sizes 32 64 \
            --results_dir "${RESULTS_DIR}/ablation_E"
        ;;

    F)
        # F needs a checkpoint — use whatever A saved, or fall back to smoke ckpt
        SMOKE_CKPT="${CKPT_DIR}"
        if [ ! -d "${RESULTS_DIR}/ablation_A/strategy_per_channel" ]; then
            echo "  [INFO] No ablation A checkpoint found; running a quick pre-train first..."
            python3 experiments/ablation_A_mv_rp.py \
                --config "${SMOKE_CFG}" \
                --train_epochs ${TRAIN_EPOCHS} \
                --probe_epochs 1 \
                --pretrain_data_dir "${ETT_DIR}" \
                --data_dir "${ICML_DIR}/ETT-small" \
                --strategies per_channel \
                --results_dir "${RESULTS_DIR}/ablation_A" 2>&1 || true
            SMOKE_CKPT="${RESULTS_DIR}/ablation_A/strategy_per_channel"
        else
            SMOKE_CKPT="${RESULTS_DIR}/ablation_A/strategy_per_channel"
        fi

        run_ablation F experiments/ablation_F_manifold.py \
            --config "${SMOKE_CFG}" \
            --checkpoint_dir "${SMOKE_CKPT}" \
            --data_dir "${ICML_DIR}/ETT-small" \
            --results_dir "${RESULTS_DIR}/ablation_F"
        ;;

    *)
        echo "Unknown ablation: ${abl}"
        ;;
    esac
done

echo ""
echo "========================================================"
echo "  Smoke test summary: ${pass} passed, ${fail} failed"
echo "  Results: ${RESULTS_DIR}"
echo "========================================================"

[ ${fail} -eq 0 ]
