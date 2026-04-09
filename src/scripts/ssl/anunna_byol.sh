#!/bin/bash
# BYOL encoder training — temporal or visual branch.
# Usage: sbatch anunna_byol.sh temporal   OR   sbatch anunna_byol.sh visual
#        MODE is passed as the first SLURM argument (--export=MODE=temporal).
# If MODE is unset, defaults to "temporal".
#SBATCH --comment=byol_train
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/byol/train_%j.out
#SBATCH --error=logs/byol/train_%j.err
#SBATCH --job-name=byol_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

MODE="${MODE:-temporal}"   # temporal | visual
mkdir -p logs/byol

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Starting BYOL ${MODE} encoder training..."
echo "Config: lotsa_byol_${MODE}.yaml"

time python3 "${SRC}/byol_training.py" \
    --config "${SRC}/configs/lotsa_byol_${MODE}.yaml" \
    --mode "${MODE}"
