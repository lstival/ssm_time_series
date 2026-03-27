#!/bin/bash
#SBATCH --comment=zero_shot_ICML
#SBATCH --time=60
#SBATCH --mem=80000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=zero_shot_inference
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/down_tasks/forecast.py
time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/evaluation_down_tasks/ICML_zeroshot_forecast.py

# how to call the job
# sbatch anunna_train_forecast.sh