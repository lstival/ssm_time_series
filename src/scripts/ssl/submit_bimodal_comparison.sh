#!/bin/bash
# Submit all 4 bimodal SSL training jobs in parallel, then probe jobs as dependencies.
#
# Usage (from repo root):
#   cd /home/WUR/stiva001/WUR/ssm_time_series
#   bash src/scripts/ssl/submit_bimodal_comparison.sh
#
# Methods: CLIP-nano, GRAM-nano, VL-JEPA-nano, SimCLR-bimodal
# Each probe job runs only after its training job succeeds (afterok).

set -euo pipefail

REPO=/home/WUR/stiva001/WUR/ssm_time_series
SCRIPTS=${REPO}/src/scripts/ssl

cd "${REPO}"

echo "=== Submitting bimodal comparison training jobs ==="

JOB_CLIP=$(sbatch --parsable "${SCRIPTS}/anunna_clip_nano.sh")
echo "  CLIP nano      → job ${JOB_CLIP}"

JOB_GRAM=$(sbatch --parsable "${SCRIPTS}/anunna_gram_nano.sh")
echo "  GRAM nano      → job ${JOB_GRAM}"

JOB_VLJEPA=$(sbatch --parsable "${SCRIPTS}/anunna_vl_jepa_nano.sh")
echo "  VL-JEPA nano   → job ${JOB_VLJEPA}"

JOB_SIMCLR=$(sbatch --parsable "${SCRIPTS}/anunna_simclr_bimodal.sh")
echo "  SimCLR bimodal → job ${JOB_SIMCLR}"

echo ""
echo "=== Submitting probe jobs (dependency: afterok on each train job) ==="

PROBE_CLIP=$(sbatch --parsable --dependency=afterok:${JOB_CLIP} "${SCRIPTS}/anunna_probe_clip_nano.sh")
echo "  Probe CLIP nano      → job ${PROBE_CLIP}  (after ${JOB_CLIP})"

PROBE_GRAM=$(sbatch --parsable --dependency=afterok:${JOB_GRAM} "${SCRIPTS}/anunna_probe_gram_nano.sh")
echo "  Probe GRAM nano      → job ${PROBE_GRAM}  (after ${JOB_GRAM})"

PROBE_VLJEPA=$(sbatch --parsable --dependency=afterok:${JOB_VLJEPA} "${SCRIPTS}/anunna_probe_vl_jepa_nano.sh")
echo "  Probe VL-JEPA nano   → job ${PROBE_VLJEPA}  (after ${JOB_VLJEPA})"

PROBE_SIMCLR=$(sbatch --parsable --dependency=afterok:${JOB_SIMCLR} "${SCRIPTS}/anunna_probe_simclr_bimodal.sh")
echo "  Probe SimCLR bimodal → job ${PROBE_SIMCLR}  (after ${JOB_SIMCLR})"

echo ""
echo "=== Summary ==="
echo "  Train  : ${JOB_CLIP} ${JOB_GRAM} ${JOB_VLJEPA} ${JOB_SIMCLR}"
echo "  Probe  : ${PROBE_CLIP} ${PROBE_GRAM} ${PROBE_VLJEPA} ${PROBE_SIMCLR}"
echo ""
echo "Results will appear in:"
echo "  results/probe_clip_nano/probe_lotsa_results.csv"
echo "  results/probe_gram_nano/probe_lotsa_results.csv"
echo "  results/probe_vl_jepa_nano/probe_lotsa_results.csv"
echo "  results/probe_simclr_bimodal_nano/probe_lotsa_results.csv"
