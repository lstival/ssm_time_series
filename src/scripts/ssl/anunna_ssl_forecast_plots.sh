#!/bin/bash
#SBATCH --comment=ssl_forecast_plots
#SBATCH --time=240
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ssl_forecast_plots/plot_%j.out
#SBATCH --error=logs/ssl_forecast_plots/plot_%j.err
#SBATCH --job-name=ssl_plots
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ssl_forecast_plots

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Generating SSL forecast comparison plots..."

time python3 "${SRC}/experiments/plot_ssl_comparison.py" \
    --horizons 96 192 336 720 \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate traffic weather \
    --n_examples 3 \
    --probe_epochs 20 \
    --context_length 96 \
    --out_dir results/ssl_forecast_plots_v2
