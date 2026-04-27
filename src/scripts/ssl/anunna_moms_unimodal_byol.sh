#!/bin/bash
#SBATCH --comment=moms_uni_byol
#SBATCH --time=720
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/uni_byol_%j.out
#SBATCH --error=logs/moms_full/uni_byol_%j.err
#SBATCH --job-name=moms_uni_byol
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full results/moms_unimodal_pipeline

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

ENCODER="byol_uni"
CKPT_DIR="checkpoints/byol_temporal/ts_byol_temporal_lotsa_20260408_1223"
CONFIG="/home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_byol_temporal.yaml"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_unimodal_pipeline"
PY="/home/WUR/stiva001/WUR/ssm_time_series/src/experiments"

echo "============================================================"
echo "MoMS Unimodal Pipeline — BYOL Temporal (Full)"
echo "Stage 1: Zero-Shot (LOTSA+Chronos → ICML)"
echo "============================================================"
time python3 ${PY}/mop_full_zeroshot.py \
  --encoder_name    "${ENCODER}" \
  --checkpoint_dir  "${CKPT_DIR}" \
  --config          "${CONFIG}" \
  --icml_data_dir   /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir     "${RESULTS}" \
  --epochs          50 --batch_size 64 --lr 1e-3 \
  --hidden_dim      512 --num_prompts 16 \
  --context_length  336 --batches_per_epoch 500 --num_workers 4 \
  --unimodal

STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}_checkpoint.pt"

echo ""
echo "============================================================"
echo "Stage 2: Few-Shot (5% ICML per dataset)"
echo "============================================================"
time python3 ${PY}/mop_full_fewshot.py \
  --encoder_name       "${ENCODER}" \
  --checkpoint_dir     "${CKPT_DIR}" \
  --mop_checkpoint     "${STAGE1_CKPT}" \
  --config             "${CONFIG}" \
  --icml_data_dir      /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir        "${RESULTS}" \
  --few_shot_fraction  0.05 --finetune_epochs 20 \
  --batch_size 64 --lr 5e-4 --hidden_dim 512 --num_prompts 16 \
  --context_length 336 --num_workers 0 \
  --unimodal

echo ""
echo "============================================================"
echo "Stage 3: GIFT-Eval fine-tune"
echo "============================================================"
time python3 ${PY}/mop_full_gift.py \
  --encoder_name        "${ENCODER}" \
  --checkpoint_dir      "${CKPT_DIR}" \
  --mop_checkpoint_dir  "${RESULTS}" \
  --stage1_checkpoint   "${STAGE1_CKPT}" \
  --config              "${CONFIG}" \
  --results_dir         "${RESULTS}" \
  --few_shot_fraction   0.10 --finetune_epochs 20 \
  --batch_size 64 --lr 5e-4 --hidden_dim 512 --num_prompts 16 \
  --context_length 336 --num_workers 0 \
  --unimodal

EXIT_CODE=$?
echo "MoMS BYOL-Unimodal done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
