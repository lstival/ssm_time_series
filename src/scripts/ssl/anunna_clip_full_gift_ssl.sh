#!/bin/bash
#SBATCH --comment=clip_full_gift_ssl
#SBATCH --time=600
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/clip_gift_ssl_%j.out
#SBATCH --error=logs/moms_full/clip_gift_ssl_%j.err
#SBATCH --job-name=clip_gift_ssl
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full
mkdir -p results/moms_clip_gift_ssl

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
ENCODER="clip"

# Resume from best CLIP-Full checkpoint (LOTSA+LOCAL pre-trained)
BASE_CKPT_DIR="checkpoints/clip_full/ts_clip_full_lotsa_20260415_012856"

CONFIG="${SRC}/configs/lotsa_clip_full_gift_ssl.yaml"
MOMS_CONFIG="${SRC}/configs/lotsa_clip_full.yaml"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_clip_gift_ssl"

echo "============================================================"
echo "CLIP-Full + GIFT-SSL Encoder Training"
echo "Corpus: LOTSA + LOCAL + Chronos + GIFT-Eval train splits"
echo "50 epochs, lr=5e-5 (fine-tune from CLIP-Full checkpoint)"
echo "Goal: check if GIFT domain data boosts zero-shot on ICML"
echo "============================================================"

# ── Encoder fine-tune on expanded corpus (adds GIFT-Eval train) ────────────
echo ""
echo "============================================================"
echo "Encoder Training: LOTSA + LOCAL + Chronos + GIFT-SSL"
echo "============================================================"
time python3 "${SRC}/simclr_bimodal_training.py" \
    --config "${CONFIG}" \
    --resume-checkpoint "${BASE_CKPT_DIR}"

# The fine-tuned encoder is saved inside BASE_CKPT_DIR (resume writes back)
FT_CKPT_DIR="${BASE_CKPT_DIR}"
echo "Fine-tuned encoder saved to: ${FT_CKPT_DIR}"

# ── Stage 1: Zero-Shot MoMS head on ICML ──────────────────────────────────
echo ""
echo "============================================================"
echo "Stage 1: Zero-Shot MoMS head (ICML datasets)"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_zeroshot.py" \
  --encoder_name    "${ENCODER}" \
  --checkpoint_dir  "${FT_CKPT_DIR}" \
  --config          "${MOMS_CONFIG}" \
  --icml_data_dir   /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir     "${RESULTS}" \
  --epochs          50 \
  --batch_size      64 \
  --lr              1e-3 \
  --hidden_dim      512 \
  --num_prompts     16 \
  --context_length  336 \
  --batches_per_epoch 500 \
  --num_workers     4

STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}_checkpoint.pt"

# ── Stage 2: Few-Shot 5% per ICML dataset ─────────────────────────────────
echo ""
echo "============================================================"
echo "Stage 2: Few-Shot 5% per ICML dataset"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_fewshot.py" \
  --encoder_name       "${ENCODER}" \
  --checkpoint_dir     "${FT_CKPT_DIR}" \
  --mop_checkpoint     "${STAGE1_CKPT}" \
  --config             "${MOMS_CONFIG}" \
  --icml_data_dir      /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir        "${RESULTS}" \
  --few_shot_fraction  0.05 \
  --finetune_epochs    20 \
  --batch_size         64 \
  --lr                 5e-4 \
  --hidden_dim         512 \
  --num_prompts        16 \
  --context_length     336 \
  --num_workers        0

EXIT_CODE=$?
echo ""
echo "CLIP-Full+GIFT-SSL pipeline done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
