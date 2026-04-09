#!/bin/bash
# Submit full ICML-v3 pipeline with dependencies:
# smoke -> training -> probe for VL-JEPA and GRAM.

set -euo pipefail

REPO=/home/WUR/stiva001/WUR/ssm_time_series
SCRIPTS=${REPO}/src/scripts/ssl

cd "${REPO}"

echo "Submitting smoke jobs..."
SMOKE_VL_JEPA=$(sbatch --parsable "${SCRIPTS}/anunna_smoke_vl_jepa.sh")
SMOKE_GRAM=$(sbatch --parsable "${SCRIPTS}/anunna_smoke_gram.sh")

echo "Submitting training jobs with afterok smoke dependencies..."
TRAIN_VL_JEPA=$(sbatch --parsable --dependency=afterok:${SMOKE_VL_JEPA} "${SCRIPTS}/anunna_vl_jepa.sh")
TRAIN_GRAM=$(sbatch --parsable --dependency=afterok:${SMOKE_GRAM} "${SCRIPTS}/anunna_gram.sh")

echo "Submitting probe jobs with afterok training dependencies..."
PROBE_VL_JEPA=$(sbatch --parsable --dependency=afterok:${TRAIN_VL_JEPA} "${SCRIPTS}/anunna_probe_vl_jepa.sh")
PROBE_GRAM=$(sbatch --parsable --dependency=afterok:${TRAIN_GRAM} "${SCRIPTS}/anunna_probe_gram.sh")

echo "========================================================"
echo "Pipeline submitted"
echo "SMOKE_VL_JEPA=${SMOKE_VL_JEPA}"
echo "SMOKE_GRAM=${SMOKE_GRAM}"
echo "TRAIN_VL_JEPA=${TRAIN_VL_JEPA}"
echo "TRAIN_GRAM=${TRAIN_GRAM}"
echo "PROBE_VL_JEPA=${PROBE_VL_JEPA}"
echo "PROBE_GRAM=${PROBE_GRAM}"
echo "========================================================"
