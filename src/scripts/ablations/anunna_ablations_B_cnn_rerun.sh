#!/bin/bash
#SBATCH --comment=ablation_B_cnn_rerun
#SBATCH --time=600
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_B/ablation_B_cnn_rerun_%j.out
#SBATCH --error=logs/ablation_B/ablation_B_cnn_rerun_%j.err
#SBATCH --job-name=ablation_B_cnn_rerun
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_B

# Ablation B — CNN-only re-run with parameter-matched architecture (~3.54M)
# Runs only sep_cnn_only variant; other variants loaded from existing checkpoints
# or will be run separately in the full ablation_B job.

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export HF_DATASETS_CACHE="/home/WUR/stiva001/.cache/hf_ablation_${SLURM_JOB_ID}"

REPO=/home/WUR/stiva001/WUR/ssm_time_series
SRC="${REPO}/src"
RESULTS="${REPO}/results/ablation_B"
CKPT="${RESULTS}/checkpoints"

time python3 "${SRC}/experiments/ablation_B_encoder_arch.py" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --train_epochs 20 \
    --probe_epochs 20 \
    --variants sep_cnn_only \
    --results_dir "${RESULTS}" \
    --checkpoint_dir "${CKPT}" \
    --data_dir "${REPO}/ICML_datasets" \
    --seed 42

# sbatch src/scripts/ablations/anunna_ablations_B_cnn_rerun.sh
