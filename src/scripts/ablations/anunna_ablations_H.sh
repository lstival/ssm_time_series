#!/bin/bash
#SBATCH --comment=ablation_H_visual_encoder_arch
#SBATCH --time=1440
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_H/ablation_H_%j.out
#SBATCH --error=logs/ablation_H/ablation_H_%j.err
#SBATCH --job-name=ablation_H_visual_encoder_arch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_H

# Ablation H — Visual Encoder Architecture
# Compares: cnn, flatten_mamba, rp_ss2d_2, ss2d_4
# Patch lengths: 16, 32, 64

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# Unique per-job HF cache
export HF_DATASETS_CACHE="/home/WUR/stiva001/.cache/hf_ablation_H_${SLURM_JOB_ID}"

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

time python3 "${SRC}/experiments/ablation_H_visual_encoder_arch.py" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --train_epochs 20 \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_H \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --seed 42

# sbatch src/scripts/anunna_ablations_H.sh
