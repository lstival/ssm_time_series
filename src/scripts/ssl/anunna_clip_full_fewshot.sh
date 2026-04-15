#!/bin/bash
#SBATCH --comment=clip_full_fewshot
#SBATCH --time=720
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clip_full/fewshot_%j.out
#SBATCH --error=logs/clip_full/fewshot_%j.err
#SBATCH --job-name=clip_full_fs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

# Few-shot linear probe for CLIP full (LOTSA pre-trained).
# Runs 1% and 5% subsampling of the ICML training set — encoder stays frozen.
# Designed to run after anunna_clip_full_probe.sh (job 66396005).
#
# Submit with dependency:
#   sbatch --dependency=afterok:66396005 src/scripts/ssl/anunna_clip_full_fewshot.sh

mkdir -p logs/clip_full

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/clip_full
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_clip_full_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No CLIP full checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Few-shot probe — CLIP FULL checkpoint: ${CHECKPOINT_DIR}"
echo "Fractions: 1% and 5% of ICML training data | encoder frozen"

RESULTS=/home/WUR/stiva001/WUR/ssm_time_series/results/clip_full

echo ""
echo "=== 1% few-shot ==="
time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_clip_full.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 30 \
    --batch_size 32 \
    --results_dir "${RESULTS}" \
    --scaler_type standard \
    --seq_len 336 \
    --seed 42 \
    --few_shot_fraction 0.01

echo ""
echo "=== 5% few-shot ==="
time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_clip_full.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 30 \
    --batch_size 32 \
    --results_dir "${RESULTS}" \
    --scaler_type standard \
    --seq_len 336 \
    --seed 42 \
    --few_shot_fraction 0.05
