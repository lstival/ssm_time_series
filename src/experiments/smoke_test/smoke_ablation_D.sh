#!/bin/bash
#SBATCH --comment=smoke_ablation_D
#SBATCH --time=120
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/ablation_D_%j.out
#SBATCH --error=logs/smoke/ablation_D_%j.err
#SBATCH --job-name=smoke_ablD
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

time python3 "${SRC}/experiments/ablation_D_visual_repr.py" \
    --config "${SRC}/experiments/smoke_test/smoke_config.yaml" \
    --train_epochs 5 \
    --probe_epochs 10 \
    --pretrain_data_dir "${REPO}/ICML_datasets/ETT-small" \
    --data_dir "${REPO}/ICML_datasets" \
    --repr_types rp gasf mtf stft \
    --probe_datasets ETTh1.csv ETTm1.csv weather.csv traffic.csv \
    --results_dir "${REPO}/results/smoke/ablation_D" \
    --seed 42

echo "End: $(date)"
