#!/bin/bash
# Submit SimCLR single-encoder training jobs + dependent linear probe jobs.
#
# For each modality (temporal, visual):
#   1. Submit encoder training (100 epochs on LOTSA)
#   2. Submit linear probe on 9 ICML datasets, with dependency on the training job.
#
# Usage:
#   cd /home/WUR/stiva001/WUR/ssm_time_series
#   bash src/scripts/submit_simclr_jobs.sh

set -euo pipefail

REPO="/home/WUR/stiva001/WUR/ssm_time_series"
SCRIPTS="${REPO}/src/scripts/ssl"

echo "========================================================"
echo "  Submitting SimCLR single-encoder jobs"
echo "========================================================"

# ── Temporal ──────────────────────────────────────────────────────────────────

echo ""
echo "[1/4] Submitting SimCLR temporal encoder training..."
TEMPORAL_TRAIN_JOB=$(MODE=temporal sbatch --parsable --export=ALL,MODE=temporal "${SCRIPTS}/anunna_simclr.sh")
echo "      Job ID: ${TEMPORAL_TRAIN_JOB}  (simclr temporal train)"

echo "[2/4] Submitting SimCLR temporal linear probe (depends on ${TEMPORAL_TRAIN_JOB})..."
TEMPORAL_PROBE_JOB=$(
    sbatch --parsable \
        --export=ALL,MODE=temporal \
        --dependency="afterok:${TEMPORAL_TRAIN_JOB}" \
        "${SCRIPTS}/anunna_probe_simclr.sh"
)
echo "      Job ID: ${TEMPORAL_PROBE_JOB}  (probe_simclr temporal)"

# ── Visual ────────────────────────────────────────────────────────────────────

echo ""
echo "[3/4] Submitting SimCLR visual encoder training..."
VISUAL_TRAIN_JOB=$(MODE=visual sbatch --parsable --export=ALL,MODE=visual "${SCRIPTS}/anunna_simclr.sh")
echo "      Job ID: ${VISUAL_TRAIN_JOB}  (simclr visual train)"

echo "[4/4] Submitting SimCLR visual linear probe (depends on ${VISUAL_TRAIN_JOB})..."
VISUAL_PROBE_JOB=$(
    sbatch --parsable \
        --export=ALL,MODE=visual \
        --dependency="afterok:${VISUAL_TRAIN_JOB}" \
        "${SCRIPTS}/anunna_probe_simclr.sh"
)
echo "      Job ID: ${VISUAL_PROBE_JOB}  (probe_simclr visual)"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  Job summary"
echo "========================================================"
echo "  Temporal train : ${TEMPORAL_TRAIN_JOB}"
echo "  Temporal probe : ${TEMPORAL_PROBE_JOB}  (waits for ${TEMPORAL_TRAIN_JOB})"
echo "  Visual   train : ${VISUAL_TRAIN_JOB}"
echo "  Visual   probe : ${VISUAL_PROBE_JOB}  (waits for ${VISUAL_TRAIN_JOB})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/simclr/train_${TEMPORAL_TRAIN_JOB}.out"
echo "  tail -f logs/simclr/train_${VISUAL_TRAIN_JOB}.out"
echo ""
echo "Results will be written to:"
echo "  ${REPO}/results/probe_simclr_temporal/"
echo "  ${REPO}/results/probe_simclr_visual/"
