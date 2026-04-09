#!/bin/bash
#SBATCH --comment=ablation_G3_horizon_dominance
#SBATCH --time=30
#SBATCH --mem=8000
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/ablation_G3/ablation_G3_%j.out
#SBATCH --error=logs/ablation_G3/ablation_G3_%j.err
#SBATCH --job-name=ablation_G3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl

mkdir -p logs/ablation_G3

# Ablation G3 — Option B: post-process existing ablation_G_results.csv
# No GPU needed — pure CSV analysis + heatmap generation.

source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
G_RESULTS=/home/WUR/stiva001/WUR/ssm_time_series/results/ablation_G/ablation_G_results.csv

echo "Start: $(date)"
echo "Input: ${G_RESULTS}"

time python3 "${SRC}/experiments/ablation_G3_horizon_dominance.py" \
    --g_results_csv "${G_RESULTS}" \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_G3

echo "End: $(date)"

# Run immediately (no GPU needed, no checkpoint dependency):
# sbatch src/scripts/anunna_ablations_G3.sh
#
# Or after ablation_G completes (to use updated results):
# sbatch --dependency=afterok:<ablation_G_job_id> src/scripts/anunna_ablations_G3.sh
