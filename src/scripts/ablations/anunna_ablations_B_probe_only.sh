#!/bin/bash
#SBATCH --comment=ablation_B_probe_only
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_B/ablation_B_probe_only_%j.out
#SBATCH --error=logs/ablation_B/ablation_B_probe_only_%j.err
#SBATCH --job-name=ablation_B_probe_only
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_B

# Ablation B — Probe-only re-run on all 5 datasets
# Skips pre-training; loads saved encoder checkpoints from a prior full run.
# Requires: results/ablation_B/checkpoints/{variant}_encoder.pt (and _visual.pt)

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export HF_DATASETS_CACHE="/home/WUR/stiva001/.cache/hf_ablation_${SLURM_JOB_ID}"

REPO=/home/WUR/stiva001/WUR/ssm_time_series
SRC="${REPO}/src"
RESULTS="${REPO}/results/ablation_B"
CKPT="${RESULTS}/checkpoints"

time python3 "${SRC}/experiments/ablation_B_encoder_arch.py" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --probe_epochs 20 \
    --results_dir "${RESULTS}" \
    --checkpoint_dir "${CKPT}" \
    --data_dir "${REPO}/ICML_datasets" \
    --probe_only \
    --seed 42

# sbatch src/scripts/ablations/anunna_ablations_B_probe_only.sh
