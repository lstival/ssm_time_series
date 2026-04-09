#!/bin/bash
# Submit lookback horizon ablation: train 5 CLIP/nano encoders then probe each.
# Usage: bash submit_lookback_ablation.sh

set -e
cd /home/WUR/stiva001/WUR/ssm_time_series

TRAIN_JOB=$(sbatch --parsable src/scripts/lookback/anunna_lookback_train.sh)
echo "Submitted training array job: ${TRAIN_JOB}"

PROBE_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} src/scripts/lookback/anunna_lookback_probe.sh)
echo "Submitted probe array job:    ${PROBE_JOB} (depends on ${TRAIN_JOB})"

echo ""
echo "Pipeline submitted:"
echo "  Train:  ${TRAIN_JOB}  (5 array tasks × 50 epochs, ~8h)"
echo "  Probe:  ${PROBE_JOB}  (starts after all train tasks complete)"
echo ""
echo "Monitor: squeue -u \$USER"
