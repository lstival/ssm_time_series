#!/bin/bash
#SBATCH --job-name=plot_probe_pred
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/WUR/stiva001/WUR/ssm_time_series/logs/plot_probe_pred_%j.out
#SBATCH --error=/home/WUR/stiva001/WUR/ssm_time_series/logs/plot_probe_pred_%j.err

set -euo pipefail

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

source /home/WUR/stiva001/WUR/timeseries/bin/activate

cd /home/WUR/stiva001/WUR/ssm_time_series

mkdir -p logs results/probe_lotsa_ablation_best

python src/experiments/plot_probe_predictions.py

echo "=== Job finished: $(date) ==="
