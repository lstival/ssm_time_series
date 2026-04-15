#!/bin/bash
#SBATCH --comment=byol_bimodal_full_train
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/byol_bimodal_full/train_%j.out
#SBATCH --error=logs/byol_bimodal_full/train_%j.err
#SBATCH --job-name=byol_bimodal_full
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/byol_bimodal_full

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Training BYOL bimodal FULL — model_dim=256, depth=8, emb=128, ctx=336, combined dataset"

time python3 "${SRC}/byol_bimodal_training.py" \
    --config "${SRC}/configs/lotsa_byol_bimodal_full.yaml"
