#!/bin/bash
#SBATCH --comment=ablation_I_mv_rp
#SBATCH --time=360
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ablation_I/ablation_I_%j.out
#SBATCH --error=logs/ablation_I/ablation_I_%j.err
#SBATCH --job-name=ablation_I_mv_rp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_I

# ─────────────────────────────────────────────────────────────────────────────
# Ablation I: Advanced Multivariate RP Methods
# ─────────────────────────────────────────────────────────────────────────────
# Tests 5 state-of-the-art methods for multivariate recurrence plots:
#
#   1. Channel Stacking (per_channel_stack)
#   2. Global L2 Distance (global_l2)
#   3. Joint Recurrence Plot (jrp_hadamard)
#   4. Cross Recurrence Plot Block (crp_block)
#   5. Multi-Scale Fusion (ms_fusion_concat)
#
# Compares on ETTm1, Weather, Traffic across horizons [96, 192, 336, 720]
# ─────────────────────────────────────────────────────────────────────────────

# Setup
module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

REPO_ROOT=$(pwd)
RESULTS_DIR="${REPO_ROOT}/results/ablation_I"
CONFIG="${REPO_ROOT}/src/configs/lotsa_clip.yaml"
DATA_DIR="${REPO_ROOT}/ICML_datasets"

mkdir -p "$RESULTS_DIR"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "Ablation I: Advanced Multivariate RP Methods"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Repository: $REPO_ROOT"
echo "Config: $CONFIG"
echo "Results dir: $RESULTS_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════════════"

SRC="${REPO_ROOT}/src"

time python3 "${SRC}/experiments/ablation_I_multivariate_rp.py" \
    --config "$CONFIG" \
    --train_epochs 20 \
    --probe_epochs 30 \
    --results_dir "$RESULTS_DIR" \
    --methods channel_stacking global_l2 jrp_hadamard crp_block ms_fusion_concat \
    --device cuda \
    --seed 42

echo "════════════════════════════════════════════════════════════════════════════════"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results saved to: $RESULTS_DIR/ablation_I_multivariate_rp.csv"
echo "════════════════════════════════════════════════════════════════════════════════"
