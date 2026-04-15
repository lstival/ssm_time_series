#!/bin/bash
#SBATCH --comment=probe_clip_micro_finetune
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clip_micro_finetune/probe_%j.out
#SBATCH --error=logs/clip_micro_finetune/probe_%j.err
#SBATCH --job-name=probe_clip_micro_ft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/clip_micro_finetune

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

CHECKPOINT_DIR=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/clip_micro_finetune

echo "Probing CLIP micro finetune checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_clip_micro_finetune.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/clip_micro_finetune \
    --scaler_type standard \
    --seed 42
