#!/bin/bash
#SBATCH --comment=mop_fewshot_simclr_nano
#SBATCH --time=1200
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mop_tuning/simclr_nano_fewshot_%j.out
#SBATCH --error=logs/mop_tuning/simclr_nano_fewshot_%j.err
#SBATCH --job-name=fs_simclr
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
MOP_CHECKPOINT="/home/WUR/stiva001/WUR/ssm_time_series/results/mop_tuning_simclr_nano/mop_latest.pt"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_fewshot_probe.py \
  --base_checkpoint_dir "${CHECKPOINT_DIR}" \
  --mop_checkpoint "${MOP_CHECKPOINT}" \
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_simclr_bimodal_nano.yaml \
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_tuning_simclr_nano \
  --few_shot_fraction 0.05 \
  --finetune_epochs 10 \
  --batch_size 64 \
  --num_prompts 8 \
  --hidden_dim 512 \
  --context_length 336
