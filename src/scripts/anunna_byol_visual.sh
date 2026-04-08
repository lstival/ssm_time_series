#!/bin/bash
#SBATCH --comment=byol_visual
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/byol_visual/train_%j.out
#SBATCH --error=logs/byol_visual/train_%j.err
#SBATCH --job-name=byol_visual
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/byol_visual

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
mkdir -p "${HF_DATASETS_CACHE}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Starting BYOL visual encoder training..."
echo "Config: lotsa_byol_visual.yaml"

time python3 "${SRC}/byol_training.py" \
    --config "${SRC}/configs/lotsa_byol_visual.yaml" \
    --mode visual
