#!/bin/bash
#SBATCH --comment=hpo_train
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/hpo/train_%j.out
#SBATCH --error=logs/hpo/train_%j.err
#SBATCH --job-name=hpo_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/hpo

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CONFIG="${SRC}/configs/lotsa_hpo_best.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: HPO best config not found at ${CONFIG}"
    echo "       HPO job must complete before this runs."
    exit 1
fi

echo "Training with HPO best config: ${CONFIG}"
time python3 "${SRC}/cosine_training.py" --config "${CONFIG}"
