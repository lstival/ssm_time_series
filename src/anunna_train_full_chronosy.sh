#!/bin/bash
#SBATCH --comment=chronos_full_temporal
#SBATCH --time=4320
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=chronos_full_temporal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/cosine_training.py --resume-checkpoint /home/WUR/stiva001/WUR/ssm_time_series/checkpoints/ts_encoder_20251101_1100_ep31
time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/chronos_supervised_training.py

# how to call the job
# sbatch anunna_train_forecast.sh