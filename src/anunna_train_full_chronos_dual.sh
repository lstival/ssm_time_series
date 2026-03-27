#!/bin/bash
#SBATCH --comment=chronos_full_temporal_dual
#SBATCH --time=4320
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=chronos_full_temporal_dual
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/down_tasks_visual/full_chronos_dual_encoder.py

# how to call the job
# sbatch anunna_train_full_chronos_dual.sh