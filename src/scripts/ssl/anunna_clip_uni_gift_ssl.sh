#!/bin/bash
#SBATCH --comment=clip_uni_gift_ssl
#SBATCH --time=60
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/clip_uni_gift_ssl_%j.out
#SBATCH --error=logs/moms_full/clip_uni_gift_ssl_%j.err
#SBATCH --job-name=clip_uni_gif
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full
mkdir -p results/moms_clip_uni_gift_ssl

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
ENCODER="clip_uni"

# Resume from best CLIP-Full temporal checkpoint (LOTSA pre-trained)
BASE_CKPT_DIR="checkpoints/clip_full/ts_clip_full_lotsa_20260415_012856"

CONFIG="${SRC}/configs/lotsa_clip_full_gift_ssl.yaml"
# For unimodal SimCLR, we use the same config but specify mode CLI
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_clip_uni_gift_ssl"

echo "============================================================"
echo "Unimodal (Temporal) CLIP-Full + GIFT-SSL Pipeline"
echo "Goal: Comparison with bimodal Job 66537292"
echo "============================================================"

# ── Stage 0: Unimodal SimCLR fine-tune on expanded corpus ───────────────────
echo ""
echo "============================================================"
echo "Stage 0: Unimodal SimCLR (Temporal) LOTSA + GIFT-SSL"
echo "============================================================"
time python3 "${SRC}/simclr_training.py" \
    --config "${CONFIG}" \
    --mode temporal \
    --resume-checkpoint "${BASE_CKPT_DIR}"

# The fine-tuned temporal encoder is saved inside the same dir by default
FT_CKPT_DIR="${BASE_CKPT_DIR}"

# ── Stage 1: Zero-Shot MoMS (Unimodal) ─────────────────────────────────────
echo ""
echo "============================================================"
echo "Stage 1: Zero-Shot MoMS (ICML datasets) --UNIMODAL"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_zeroshot.py" \
  --encoder_name    "${ENCODER}" \
  --checkpoint_dir  "${FT_CKPT_DIR}" \
  --config          "${CONFIG}" \
  --icml_data_dir   /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir     "${RESULTS}" \
  --epochs          50 \
  --batch_size      64 \
  --lr              1e-3 \
  --hidden_dim      512 \
  --num_prompts     16 \
  --context_length  336 \
  --batches_per_epoch 500 \
  --num_workers     4 \
  --unimodal

STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}_checkpoint.pt"

# ── Stage 2: Few-Shot (Unimodal) ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "Stage 2: Few-Shot 5% (ICML datasets) --UNIMODAL"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_fewshot.py" \
  --encoder_name       "${ENCODER}" \
  --checkpoint_dir     "${FT_CKPT_DIR}" \
  --mop_checkpoint     "${STAGE1_CKPT}" \
  --config             "${CONFIG}" \
  --icml_data_dir      /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir        "${RESULTS}" \
  --few_shot_fraction  0.05 \
  --finetune_epochs    20 \
  --batch_size         64 \
  --lr                 5e-4 \
  --hidden_dim         512 \
  --num_prompts        16 \
  --context_length     336 \
  --num_workers        0 \
  --unimodal

# ── Stage 3: GIFT-Eval fine-tune (Unimodal) ───────────────────────────────
echo ""
echo "============================================================"
echo "Stage 3: GIFT-Eval Fine-Tune --UNIMODAL"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_gift.py" \
  --encoder_name        "${ENCODER}" \
  --checkpoint_dir      "${FT_CKPT_DIR}" \
  --mop_checkpoint_dir  "${RESULTS}" \
  --stage1_checkpoint   "${STAGE1_CKPT}" \
  --config              "${CONFIG}" \
  --results_dir         "${RESULTS}" \
  --few_shot_fraction   0.10 \
  --finetune_epochs     20 \
  --batch_size          64 \
  --lr                  5e-4 \
  --hidden_dim          512 \
  --num_prompts         16 \
  --context_length      336 \
  --num_workers         0 \
  --unimodal

EXIT_CODE=$?
echo ""
echo "Unimodal CLIP-Full GIFT-SSL pipeline done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
