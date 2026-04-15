#!/bin/bash
#SBATCH --comment=clip_hpo
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/hpo/hpo_%j.out
#SBATCH --error=logs/hpo/hpo_%j.err
#SBATCH --job-name=clip_hpo
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

echo "Starting Optuna HPO: 30 trials × 20 epochs each"

time python3 "${SRC}/experiments/optuna_hpo.py" \
    --base_config "${SRC}/configs/lotsa_best.yaml" \
    --n_trials 30 \
    --trial_epochs 20 \
    --output_config "${SRC}/configs/lotsa_hpo_best.yaml" \
    --study_name clip_hpo
