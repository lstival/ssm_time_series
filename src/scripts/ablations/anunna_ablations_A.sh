#!/bin/bash
#SBATCH --comment=ablation_A_mv_rp
#SBATCH --time=1440
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_A/ablation_A_%j.out
#SBATCH --error=logs/ablation_A/ablation_A_%j.err
#SBATCH --job-name=ablation_A_mv_rp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_A

# Ablation A — Multivariate RP Definition
# Compares: per_channel, mean, pca, joint
# Expected runtime: ~4 h on A100 (20 train epochs × 4 strategies)

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

time python3 "${SRC}/experiments/ablation_A_mv_rp.py" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --train_epochs 20 \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_A \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --seed 42

# How to submit (with dependency on LOTSA download):
# sbatch --dependency=afterok:<download_job> src/anunna_ablations_A.sh
# Standalone:
# sbatch src/scripts/anunna_ablations_A.sh
