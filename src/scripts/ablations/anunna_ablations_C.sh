#!/bin/bash
#SBATCH --comment=ablation_C_alignment
#SBATCH --time=1440
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_C/ablation_C_%j.out
#SBATCH --error=logs/ablation_C/ablation_C_%j.err
#SBATCH --job-name=ablation_C_alignment
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_C

# Ablation C — Contrastive Alignment Strategy
# Compares: clip_symm (base), cosine_mse, concat_supervised, unimodal_temporal

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# Unique per-job HF cache to avoid race conditions when jobs run concurrently
export HF_DATASETS_CACHE="/home/WUR/stiva001/.cache/hf_ablation_${SLURM_JOB_ID}"

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

time python3 "${SRC}/experiments/ablation_C_alignment.py" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --train_epochs 20 \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_C \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --seed 42

# sbatch src/scripts/anunna_ablations_C.sh
