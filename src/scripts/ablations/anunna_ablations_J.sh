#!/bin/bash
#SBATCH --comment=ablation_J_ssm_hparams
#SBATCH --time=720
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_J/ablation_J_%j.out
#SBATCH --error=logs/ablation_J/ablation_J_%j.err
#SBATCH --job-name=ablation_J_ssm_hparams
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_J

# Ablation J - Validate SSM hyperparameters on nano model.
# Baseline from paper: d_SSM=16, k=4
# Sweep smaller and larger values around the baseline.

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export HF_DATASETS_CACHE="/home/WUR/stiva001/.cache/hf_ablation_J_${SLURM_JOB_ID}"

REPO_ROOT=$(pwd)
SRC="${REPO_ROOT}/src"

time python3 "${SRC}/experiments/ablation_J_ssm_hparams.py" \
    --config "${SRC}/configs/lotsa_nano.yaml" \
    --train_epochs 20 \
    --probe_epochs 20 \
    --state_dims 8 16 32 \
    --conv_kernels 2 4 8 \
    --results_dir "${REPO_ROOT}/results/ablation_J" \
    --data_dir "${REPO_ROOT}/ICML_datasets" \
    --seed 42

# Submit with:
# sbatch src/scripts/ablations/anunna_ablations_J.sh
