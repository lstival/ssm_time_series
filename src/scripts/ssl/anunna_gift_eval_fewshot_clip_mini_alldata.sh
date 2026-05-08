#!/bin/bash
#SBATCH --comment=gift_eval_fs_mini
#SBATCH --time=720
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gift_eval/gift_eval_fewshot_mini_alldata_%j.out
#SBATCH --error=logs/gift_eval/gift_eval_fewshot_mini_alldata_%j.err
#SBATCH --job-name=gift_fs_mini
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

set -euo pipefail

mkdir -p logs/gift_eval
mkdir -p results/moms_clip_mini_alldata
mkdir -p results/gift_eval_official

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
export GIFT_EVAL="/lustre/nobackup/WUR/AIN/stiva001/gift_eval_storage"
mkdir -p "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

cd /home/WUR/stiva001/WUR/ssm_time_series

SRC=src
ENCODER="clip_mini_alldata"
CONFIG="${SRC}/configs/lotsa_clip_mini_alldata.yaml"
RESULTS="results/moms_clip_mini_alldata"
ICML_DATA="ICML_datasets"

CKPT_DIR=$(ls -td checkpoints/clip_mini_alldata/ts_clip_mini_alldata_* 2>/dev/null | head -1)
if [ -z "${CKPT_DIR}" ]; then echo "ERROR: encoder checkpoint not found"; exit 1; fi
echo "Encoder: ${CKPT_DIR}"

# Require Stage 1 zero-shot checkpoint as warm-start
STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}_checkpoint.pt"
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "ERROR: Stage 1 zero-shot checkpoint not found: ${STAGE1_CKPT}"
    echo "Run anunna_gift_eval_zeroshot_clip_mini_alldata.sh first."
    exit 1
fi

# ── Stage 2: ICML few-shot fine-tune ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "Stage 2: MoP Few-Shot fine-tune on ICML datasets"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_fewshot.py" \
    --encoder_name      "${ENCODER}" \
    --checkpoint_dir    "${CKPT_DIR}" \
    --mop_checkpoint    "${STAGE1_CKPT}" \
    --config            "${CONFIG}" \
    --icml_data_dir     "${ICML_DATA}" \
    --results_dir       "${RESULTS}" \
    --epochs            30 \
    --batch_size        64 \
    --lr                5e-4 \
    --hidden_dim        512 \
    --num_prompts       16 \
    --context_length    336 \
    --num_workers       0

STAGE2_CKPT="${RESULTS}/mop_fewshot_${ENCODER}_checkpoint.pt"
if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "ERROR: Stage 2 checkpoint not found: ${STAGE2_CKPT}"
    exit 1
fi

# ── Stage 3: GIFT-Eval few-shot fine-tune + evaluation ───────────────────────
echo ""
echo "============================================================"
echo "Stage 3: GIFT-Eval few-shot fine-tune — CLIP mini alldata"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_gift.py" \
    --encoder_name          "${ENCODER}" \
    --checkpoint_dir        "${CKPT_DIR}" \
    --mop_checkpoint_dir    "${RESULTS}" \
    --stage1_checkpoint     "${STAGE1_CKPT}" \
    --config                "${CONFIG}" \
    --results_dir           "${RESULTS}" \
    --finetune_epochs       30 \
    --batch_size            64 \
    --lr                    1e-4 \
    --hidden_dim            512 \
    --num_prompts           16 \
    --context_length        336 \
    --num_workers           0

GIFT_CKPT="${RESULTS}/mop_gift_${ENCODER}_checkpoint.pt"
if [ ! -f "${GIFT_CKPT}" ]; then
    # Fallback: mop_full_gift may save per-subset checkpoints; use stage1 for eval
    GIFT_CKPT="${STAGE1_CKPT}"
    echo "WARNING: per-subset gift checkpoint not found, using zero-shot checkpoint for eval"
fi

# ── Stage 4: GIFT-Eval Official Evaluation (few-shot) ────────────────────────
echo ""
echo "============================================================"
echo "Stage 4: GIFT-Eval NRMSE/SMAPE Evaluation — few-shot"
echo "============================================================"
time python3 "${SRC}/experiments/gift_eval_official_eval.py" \
    --encoder_name      "${ENCODER}" \
    --checkpoint_dir    "${CKPT_DIR}" \
    --mop_checkpoint    "${GIFT_CKPT}" \
    --config            "${CONFIG}" \
    --mode              fewshot \
    --results_dir       "results/gift_eval_official/${ENCODER}_fewshot" \
    --context_length    336 \
    --batch_size        64

EXIT_CODE=$?
echo "Done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
