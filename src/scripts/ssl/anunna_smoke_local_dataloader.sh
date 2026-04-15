#!/bin/bash
#SBATCH --comment=smoke_local_dataloader
#SBATCH --time=00:30:00
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke_local_dataloader/smoke_%j.out
#SBATCH --error=logs/smoke_local_dataloader/smoke_%j.err
#SBATCH --job-name=smoke_local_dl
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --partition=main

mkdir -p logs/smoke_local_dataloader

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"
source /home/WUR/stiva001/WUR/timeseries/bin/activate

REPO=/home/WUR/stiva001/WUR/ssm_time_series
cd "${REPO}"

echo "=== Smoke test: local_dataset_loader ==="
PYTHONPATH="${REPO}/src" time python3 src/dataloaders/local_dataset_loader.py
