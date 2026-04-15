#!/bin/bash
#SBATCH --comment=clip_full_train
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clip_full/train_%j.out
#SBATCH --error=logs/clip_full/train_%j.err
#SBATCH --job-name=clip_full
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/clip_full

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Training CLIP bimodal FULL — model_dim=256, depth=8, emb=128, ctx=512, HPO hyperparams, combined dataset"

time python3 "${SRC}/simclr_bimodal_training.py" \
    --config "${SRC}/configs/lotsa_clip_full.yaml"
