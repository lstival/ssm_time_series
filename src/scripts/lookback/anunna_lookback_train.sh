#!/bin/bash
#SBATCH --comment=lookback_train
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/lookback/train_%a_%j.out
#SBATCH --error=logs/lookback/train_%a_%j.err
#SBATCH --job-name=lookback_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'
#SBATCH --array=0-4

mkdir -p logs/lookback

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

LOOKBACKS=(96 192 336 512 720)
CTX=${LOOKBACKS[$SLURM_ARRAY_TASK_ID]}
CONFIG="${SRC}/configs/lotsa_lookback_${CTX}.yaml"

echo "Training lookback=${CTX}  (array task ${SLURM_ARRAY_TASK_ID})"

time python3 "${SRC}/cosine_training.py" --config "${CONFIG}"
