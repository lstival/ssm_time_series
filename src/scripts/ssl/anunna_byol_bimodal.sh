#!/bin/bash
#SBATCH --comment=byol_bimodal
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/byol_bimodal/train_%j.out
#SBATCH --error=logs/byol_bimodal/train_%j.err
#SBATCH --job-name=byol_bimodal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/byol_bimodal

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Starting BYOL bimodal training (temporal + visual RP, EMA targets)..."

time python3 "${SRC}/byol_bimodal_training.py" \
    --config "${SRC}/configs/lotsa_byol_bimodal_nano.yaml"
