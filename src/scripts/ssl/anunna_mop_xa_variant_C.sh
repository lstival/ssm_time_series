#!/bin/bash
#SBATCH --comment=mop_xa_varC
#SBATCH --time=480
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gift_eval/mop_xa_varC_%j.out
#SBATCH --error=logs/gift_eval/mop_xa_varC_%j.err
#SBATCH --job-name=xa_varC
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

# Variant C: shared projection space + cosine alignment loss.
# Temporal and visual embeddings projected to shared space and summed;
# alignment loss (alpha=0.1) encourages modality consistency.

set -euo pipefail

mkdir -p logs/gift_eval results/moms_clip_mini_alldata results/gift_eval_official

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export LOTSA_CACHE_DIR="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
export GIFT_EVAL="/lustre/nobackup/WUR/AIN/stiva001/gift_eval_storage"
mkdir -p "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

cd /lustre/nobackup/WUR/AIN/stiva001/ssm_time_series

SRC=src
ENCODER="clip_mini_alldata"
CONFIG="${SRC}/configs/lotsa_clip_mini_alldata.yaml"
RESULTS="results/moms_clip_mini_alldata"
ICML_DATA="ICML_datasets"
VARIANT="C"
SUFFIX="_xa_varC"

CKPT_DIR=$(ls -td checkpoints/clip_mini_alldata/ts_clip_mini_alldata_* 2>/dev/null | head -1)
if [ -z "${CKPT_DIR}" ]; then echo "ERROR: encoder checkpoint not found"; exit 1; fi
echo "Encoder: ${CKPT_DIR}"

STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}${SUFFIX}_checkpoint.pt"

echo "============================================================"
echo "Stage 1: MoP training — cross-attn variant ${VARIANT}"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_zeroshot.py" \
    --encoder_name        "${ENCODER}" \
    --checkpoint_dir      "${CKPT_DIR}" \
    --config              "${CONFIG}" \
    --icml_data_dir       "${ICML_DATA}" \
    --results_dir         "${RESULTS}" \
    --epochs              100 \
    --batch_size          64 \
    --lr                  3e-4 \
    --hidden_dim          512 \
    --num_prompts         16 \
    --context_length      336 \
    --batches_per_epoch   500 \
    --num_workers         0 \
    --mop_crossattn \
    --crossattn_variant   "${VARIANT}" \
    --crossattn_heads     4 \
    --align_alpha         0.1 \
    --output_suffix       "${SUFFIX}"

echo "============================================================"
echo "Stage 2: GIFT-Eval evaluation — variant ${VARIANT}"
echo "============================================================"
time python3 "${SRC}/experiments/gift_eval_official_eval.py" \
    --encoder_name      "${ENCODER}" \
    --checkpoint_dir    "${CKPT_DIR}" \
    --mop_checkpoint    "${STAGE1_CKPT}" \
    --config            "${CONFIG}" \
    --mode              zeroshot \
    --results_dir       "results/gift_eval_official/${ENCODER}_xa_var${VARIANT}" \
    --context_length    336 \
    --batch_size        64

EXIT_CODE=$?
echo "Done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
