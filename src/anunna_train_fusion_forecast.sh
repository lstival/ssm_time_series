#!/bin/bash
#SBATCH --comment=chronos_forecast
#SBATCH --time=1200
#SBATCH --mem=30000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=forecast_chronos.py
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/down_tasks/forecast.py
time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/down_tasks_visual/forecast_chronos_frozen_dual.py

# how to call the job
# sbatch forecast_chronos_frozen_dual.sh