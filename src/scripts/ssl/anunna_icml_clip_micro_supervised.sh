#!/bin/bash
#SBATCH --comment=icml_clip_micro_supervised
#SBATCH --time=720
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icml_clip_micro/supervised_%j.out
#SBATCH --error=logs/icml_clip_micro/supervised_%j.err
#SBATCH --job-name=icml_clip_sup
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/icml_clip_micro

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Supervised training (random-init encoders) on ICML datasets — micro config"

time python3 "${SRC}/experiments/icml_finetune_vs_supervised.py" \
    --mode supervised \
    --config "${SRC}/configs/icml_clip_micro.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/icml_clip_micro_supervised \
    --epochs 50 \
    --batch_size 128 \
    --lr 1e-4 \
    --encoder_lr_scale 0.1 \
    --mlp_hidden_dim 512 \
    --context_length 336 \
    --horizons 96 192 336 720 \
    --scaler_type standard
