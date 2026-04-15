#!/bin/bash
#SBATCH --comment=clip_micro_finetune
#SBATCH --time=1440
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clip_micro_finetune/train_%j.out
#SBATCH --error=logs/clip_micro_finetune/train_%j.err
#SBATCH --job-name=clip_micro_ft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/clip_micro_finetune

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
RESUME=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/hpo_best/ts_hpo_best_lotsa_20260411_193513

echo "Finetuning CLIP micro from HPO best checkpoint..."
echo "Resume: ${RESUME}"
echo "Dataset: combined (LOTSA + local financial/energy/mobility)"
echo "Epochs: 50 additional | lr=1e-4 | ctx=512"

CKPT_OUT=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/clip_micro_finetune

time python3 "${SRC}/cosine_training.py" \
    --config "${SRC}/configs/lotsa_clip_micro_finetune.yaml" \
    --resume-checkpoint "${RESUME}" \
    --checkpoint-dir "${CKPT_OUT}"
