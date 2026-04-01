#!/bin/bash
#SBATCH --comment=forecast_optimized_v2
#SBATCH --time=1200
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/forecast_optimized_v2/train_forecast_optimized_v2_%j.out
#SBATCH --error=logs/forecast_optimized_v2/train_forecast_optimized_v2_%j.err
#SBATCH --job-name=forecast_opt_v2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/forecast_optimized_v2

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# Train forecast head using encoder from lotsa_optimized_v2
# Will automatically load the best encoder checkpoint from the encoder training job
time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/down_tasks/forecast_chronos.py

# Job: ICML v2 forecast head training (follows encoder job)
#
# Uses:
# - Encoder from: lotsa_optimized_v2 (job 66150723 after completion)
# - Ablation optimizations: joint RP strategy (25% faster)
#
# Expected behavior:
# 1. Loads best encoder checkpoint from lotsa_optimized_v2
# 2. Trains multi-horizon forecast head on ICML datasets
# 3. Evaluates on ETTm1, weather, exchange_rate, etc.
#
# Notes:
# - This job should start after encoder job 66150723 completes
# - Checkpoint paths may need adjustment if encoder saves to different location
#
# To submit with dependency on encoder job:
# sbatch --dependency=afterok:66150723 src/scripts/anunna_train_forecast_optimized_v2.sh
