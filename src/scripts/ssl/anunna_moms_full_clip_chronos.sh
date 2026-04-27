#!/bin/bash
#SBATCH --comment=moms_full_clip_chronos
#SBATCH --time=720
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/clip_chronos_%j.out
#SBATCH --error=logs/moms_full/clip_chronos_%j.err
#SBATCH --job-name=moms_clip_ch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full
mkdir -p results/moms_full_chronos

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
ENCODER="clip"
BASE_CKPT_DIR="checkpoints/clip_full/ts_clip_full_lotsa_20260415_012856"
FT_CKPT_DIR="checkpoints/clip_full_chronos_ft"
FT_CONFIG="${SRC}/configs/lotsa_clip_full_chronos_ft.yaml"
MOMS_CONFIG="${SRC}/configs/lotsa_clip_full.yaml"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_chronos"

echo "============================================================"
echo "MoMS Full Pipeline — CLIP-Full + Chronos corpus"
echo "Encoder fine-tune: LOTSA + LOCAL + Chronos (30 epochs, lr=5e-5)"
echo "Stage 1: Zero-Shot MoMS head (ICML)"
echo "Stage 2: Few-Shot 5% per dataset (ICML)"
echo "Stage 3: GIFT-Eval fine-tune"
echo "============================================================"

# ── Encoder fine-tune on expanded corpus ───────────────────────────────────
echo ""
echo "============================================================"
echo "Encoder Fine-Tune: LOTSA + LOCAL + Chronos"
echo "============================================================"
time python3 "${SRC}/simclr_bimodal_training.py" \
    --config "${FT_CONFIG}" \
    --resume-checkpoint "${BASE_CKPT_DIR}"

# Resume saves back into the same checkpoint dir as the base
FT_CKPT_DIR_RESOLVED="${BASE_CKPT_DIR}"
echo "Fine-tuned encoder: ${FT_CKPT_DIR_RESOLVED}"

# ── Stage 1: Zero-Shot MoMS head ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "Stage 1: Zero-Shot MoMS head (ICML datasets)"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_zeroshot.py" \
  --encoder_name    "${ENCODER}" \
  --checkpoint_dir  "${FT_CKPT_DIR_RESOLVED}" \
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

# ── Stage 2: Few-Shot 5% fine-tune per ICML dataset ────────────────────────
echo ""
echo "============================================================"
echo "Stage 2: Few-Shot 5% per ICML dataset"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_fewshot.py" \
  --encoder_name       "${ENCODER}" \
  --checkpoint_dir     "${FT_CKPT_DIR_RESOLVED}" \
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

# ── Stage 3: GIFT-Eval fine-tune ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "Stage 3: GIFT-Eval fine-tune"
echo "============================================================"
time python3 "${SRC}/experiments/mop_full_gift.py" \
  --encoder_name        "${ENCODER}" \
  --checkpoint_dir      "${FT_CKPT_DIR_RESOLVED}" \
  --mop_checkpoint_dir  "${RESULTS}" \
  --stage1_checkpoint   "${STAGE1_CKPT}" \
  --config              "${MOMS_CONFIG}" \
  --results_dir         "${RESULTS}" \
  --few_shot_fraction   0.10 \
  --finetune_epochs     20 \
  --batch_size          64 \
  --lr                  5e-4 \
  --hidden_dim          512 \
  --num_prompts         16 \
  --context_length      336 \
  --num_workers         0

EXIT_CODE=$?
echo ""
echo "MoMS CLIP+Chronos pipeline done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
