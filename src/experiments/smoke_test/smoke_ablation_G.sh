#!/bin/bash
#SBATCH --comment=smoke_ablation_G
#SBATCH --time=60
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/ablation_G_%j.out
#SBATCH --error=logs/smoke/ablation_G_%j.err
#SBATCH --job-name=smoke_ablG
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/smoke

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
REPO=/home/WUR/stiva001/WUR/ssm_time_series

echo "Start: $(date)"
echo "Mode: smoke test (random weights, ETTm1 only, H=96+192, 2 probe epochs)"

time python3 "${SRC}/experiments/ablation_G_encoder_modes.py" \
    --random_weights \
    --config "${SRC}/experiments/smoke_test/smoke_config.yaml" \
    --data_dir "${REPO}/ICML_datasets" \
    --datasets ETTm1.csv \
    --horizons 96 192 \
    --probe_epochs 2 \
    --encoder_modes temporal_only visual_only multimodal \
    --results_dir "${REPO}/results/smoke/ablation_G" \
    --no_comet \
    --seed 42

echo "End: $(date)"
