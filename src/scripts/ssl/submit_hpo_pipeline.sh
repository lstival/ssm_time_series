#!/bin/bash
# Submit full HPO pipeline: HPO search → full train → linear probe
# Usage: bash submit_hpo_pipeline.sh

set -e
cd /home/WUR/stiva001/WUR/ssm_time_series

mkdir -p logs/hpo

HPO_JOB=$(sbatch --parsable src/scripts/ssl/anunna_hpo.sh)
echo "Submitted HPO job:   ${HPO_JOB}  (30 trials × 20 epochs)"

TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${HPO_JOB} src/scripts/ssl/anunna_hpo_train.sh)
echo "Submitted train job: ${TRAIN_JOB}  (100 epochs with best config)"

PROBE_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} src/scripts/ssl/anunna_hpo_probe.sh)
echo "Submitted probe job: ${PROBE_JOB}  (linear probe on ICML datasets)"

echo ""
echo "Pipeline:"
echo "  HPO search : ${HPO_JOB}   (~48h, writes src/configs/lotsa_hpo_best.yaml)"
echo "  Full train : ${TRAIN_JOB}  (starts after HPO)"
echo "  Probe      : ${PROBE_JOB}  (starts after train)"
echo ""
echo "Results will be in: results/hpo_best/probe_lotsa_results.csv"
echo "Monitor: squeue -u \$USER"
