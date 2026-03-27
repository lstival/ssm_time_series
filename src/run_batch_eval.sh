#!/bin/bash
#SBATCH --comment=tfb_batch_eval
#SBATCH --time=4320
#SBATCH --mem=10000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=tfb_batch_eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'


module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate


time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/evaluation_tfb/batch_evaluator.py \
    --data_dir "data/forecasting/forecasting" \
    --model_config "src/configs/tfb_eval_model.yaml" \
    --forecast_checkpoint "checkpoints/multi_horizon_forecast_dual_frozen_20251209_1049/all_datasets/best_model.pt" \
    --encoder_checkpoint "checkpoints/ts_encoder_20251126_1750/time_series_best.pt" \
    --visual_encoder_checkpoint "checkpoints/ts_encoder_20251126_1750/visual_encoder_best.pt" \
    --horizons 96 192 336 720 \
    --results_dir "results/tfb_batch_full" \
    --stride 100 \
    --device cuda
