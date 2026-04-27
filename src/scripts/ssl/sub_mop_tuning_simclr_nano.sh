#!/bin/bash
#SBATCH --comment=mop_tuning_simclr_nano
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mop_tuning/simclr_nano_%j.out
#SBATCH --error=logs/mop_tuning/simclr_nano_%j.err
#SBATCH --job-name=mop_simclr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/mop_tuning
mkdir -p results/mop_tuning_simclr_nano

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

CHECKPOINT_DIR="checkpoints/simclr_bimodal_nano/ts_simclr_bimodal_nano_lotsa_20260409_182831"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_tuning.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_simclr_bimodal_nano.yaml \
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_tuning_simclr_nano \
  --epochs 20 \
  --batch_size 64 \
  --num_prompts 8 \
  --hidden_dim 512 \
  --context_length 336 \
  --max_horizon 720 \
  --batches_per_epoch 1000
