#!/bin/bash
#SBATCH --comment=simclr_temporal_nano
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr_temporal_nano/train_%j.out
#SBATCH --error=logs/simclr_temporal_nano/train_%j.err
#SBATCH --job-name=simclr_t_nano
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/simclr_temporal_nano

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Training SimCLR temporal-only NANO — model_dim=128, depth=4, emb=64, ctx=336"

time python3 "${SRC}/simclr_training.py" \
    --config "${SRC}/configs/lotsa_simclr_temporal_nano.yaml" \
    --mode temporal
