#!/bin/bash
#SBATCH --comment=ablation_F_manifold
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_F/ablation_F_%j.out
#SBATCH --error=logs/ablation_F/ablation_F_%j.err
#SBATCH --job-name=ablation_F_manifold
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_F

# Ablation F — Manifold Quality on Unseen Evaluation Data
# Requires: pretrained checkpoint from cosine_training (e.g. ts_encoder_lotsa)
# Produces: t-SNE plots + Silhouette / Davies-Bouldin / Cohesion / Separation

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/home/WUR/stiva001/WUR/ssm_time_series/checkpoints

# Update CHECKPOINT_DIR to the trained lotsa checkpoint directory
CHECKPOINT_DIR="${CHECKPOINTS}/ts_encoder_lotsa_20260329_2031"

time python3 "${SRC}/experiments/ablation_F_manifold.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_clip.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_F \
    --random_baseline \
    --seed 42

# Run AFTER cosine training completes:
# sbatch --dependency=afterok:<training_job> src/anunna_ablations_F.sh
# Standalone (if checkpoint already exists):
# sbatch src/scripts/anunna_ablations_F.sh
