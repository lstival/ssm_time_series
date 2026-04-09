#!/bin/bash
#SBATCH --comment=smoke_ablation_B
#SBATCH --time=120
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/ablation_B_%j.out
#SBATCH --error=logs/smoke/ablation_B_%j.err
#SBATCH --job-name=smoke_ablB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/smoke

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
REPO=/home/WUR/stiva001/WUR/ssm_time_series

echo "Start: $(date)"

time python3 "${SRC}/experiments/ablation_B_encoder_arch.py" \
    --config "${SRC}/experiments/smoke_test/smoke_config.yaml" \
    --train_epochs 5 \
    --probe_epochs 10 \
    --pretrain_data_dir "${REPO}/ICML_datasets/ETT-small" \
    --data_dir "${REPO}/ICML_datasets" \
    --variants no_visual shared_1d sep_cnn_only sep_mamba_1d \
    --results_dir "${REPO}/results/smoke/ablation_B" \
    --seed 42

echo "End: $(date)"
