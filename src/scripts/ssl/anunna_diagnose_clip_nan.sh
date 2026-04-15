#!/bin/bash
#SBATCH --comment=diagnose_clip_nan
#SBATCH --time=120
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/diagnose_clip_nan/diag_%j.out
#SBATCH --error=logs/diagnose_clip_nan/diag_%j.err
#SBATCH --job-name=clip_nan_diag
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/diagnose_clip_nan

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

REPO_ROOT=$(pwd)
SRC="${REPO_ROOT}/src"
CONFIG="${SRC}/configs/lotsa_clip_full.yaml"
RESULTS="${REPO_ROOT}/results/diagnose_clip_nan"

echo "════════════════════════════════════════════════════════════════"
echo "CLIP NaN Diagnostics — Job $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"

# ── Run 1: baseline (no AMP, no grad clip) — reproduce the NaN
echo ""
echo "▶ Run 1: No AMP, no grad clip (reproduce NaN)"
time python3 "${SRC}/experiments/diagnose_clip_nan.py" \
    --config "$CONFIG" \
    --results_dir "${RESULTS}/run1_baseline" \
    --epochs 12 \
    --max_batches 150 \
    --seed 42

# ── Run 2: with AMP (test if AMP causes the NaN via scaler overflow)
echo ""
echo "▶ Run 2: With AMP (test scaler overflow)"
time python3 "${SRC}/experiments/diagnose_clip_nan.py" \
    --config "$CONFIG" \
    --results_dir "${RESULTS}/run2_amp" \
    --epochs 12 \
    --max_batches 150 \
    --use_amp \
    --seed 42

# ── Run 3: grad clip only (test if clipping fixes NaN)
echo ""
echo "▶ Run 3: grad clip=1.0, no AMP (test if clipping prevents NaN)"
time python3 "${SRC}/experiments/diagnose_clip_nan.py" \
    --config "$CONFIG" \
    --results_dir "${RESULTS}/run3_gradclip" \
    --epochs 12 \
    --max_batches 150 \
    --grad_clip 1.0 \
    --seed 42

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results: $RESULTS"
echo "════════════════════════════════════════════════════════════════"
