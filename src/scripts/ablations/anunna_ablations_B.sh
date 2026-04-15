#!/bin/bash
#SBATCH --comment=ablation_B_encoder_arch
#SBATCH --time=1440
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_B/ablation_B_%j.out
#SBATCH --error=logs/ablation_B/ablation_B_%j.err
#SBATCH --job-name=ablation_B_encoder_arch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_B

# Ablation B — Visual Encoder Architecture
# Compares: no_visual, shared_1d, sep_cnn_only, sep_mamba_1d
# Probes on: ETTh1, ETTm1, Weather, Traffic, Solar

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# Unique per-job HF cache to avoid race conditions when jobs run concurrently
export HF_DATASETS_CACHE="/home/WUR/stiva001/.cache/hf_ablation_${SLURM_JOB_ID}"

REPO=/home/WUR/stiva001/WUR/ssm_time_series
SRC="${REPO}/src"
RESULTS="${REPO}/results/ablation_B"
CKPT="${RESULTS}/checkpoints"

time python3 "${SRC}/experiments/ablation_B_encoder_arch.py" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --train_epochs 20 \
    --probe_epochs 20 \
    --results_dir "${RESULTS}" \
    --checkpoint_dir "${CKPT}" \
    --data_dir "${REPO}/ICML_datasets" \
    --seed 42

# To re-run probes only (skip pre-training, reuse saved encoders):
#   sbatch --job-name=ablation_B_probe_only src/scripts/ablations/anunna_ablations_B.sh
# with the extra flag:
#   python3 ... --probe_only --checkpoint_dir "${CKPT}"
