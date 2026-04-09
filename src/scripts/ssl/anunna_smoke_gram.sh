#!/bin/bash
#SBATCH --comment=smoke_gram
#SBATCH --time=00:30:00
#SBATCH --mem=16000
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/smoke_gram/smoke_%j.out
#SBATCH --error=logs/smoke_gram/smoke_%j.err
#SBATCH --job-name=smoke_gram
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/smoke_gram

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

REPO=/home/WUR/stiva001/WUR/ssm_time_series
cd "${REPO}"

time python3 src/experiments/smoke_test/smoke_gram.py --epochs 3 --no_comet
