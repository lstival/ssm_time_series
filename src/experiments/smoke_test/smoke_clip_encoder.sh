#!/bin/bash
#SBATCH --comment=smoke_clip_encoder
#SBATCH --time=120
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/clip_encoder_%j.out
#SBATCH --error=logs/smoke/clip_encoder_%j.err
#SBATCH --job-name=smoke_clip
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

time python3 "${SRC}/experiments/smoke_test/smoke_clip_encoder.py" \
    --config "${SRC}/experiments/smoke_test/smoke_config.yaml" \
    --data_dir "${REPO}/ICML_datasets/ETT-small" \
    --epochs 10 \
    --seed 42

echo "End: $(date)"
