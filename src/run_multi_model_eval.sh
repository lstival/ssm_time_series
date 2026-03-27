#!/bin/bash
#SBATCH --comment=multi_model_eval
#SBATCH --time=4320
#SBATCH --mem=10000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=multi_model_eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'


module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate


time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/evaluation_tfb/multi_model_eval_runner.py \
    --data_dir "data/forecasting/forecasting/" \
    --models timesfm chronos transformer patchtst \
    --horizons 96 192 336 720
