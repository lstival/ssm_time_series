#!/bin/bash
#SBATCH --comment=moms_gift_supervised_simclr
#SBATCH --time=480
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/gift_supervised_simclr_%j.out
#SBATCH --error=logs/moms_full/gift_supervised_simclr_%j.err
#SBATCH --job-name=gift_sup_sim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full
mkdir -p results/moms_gift_supervised

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
ENCODER="simclr"
CKPT_DIR="checkpoints/simclr_full/ts_simclr_full_lotsa_20260414_002946"
CONFIG="${SRC}/configs/lotsa_simclr_full.yaml"
# Use Stage 1 checkpoint from chronos run (best available)
STAGE1_CKPT="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_chronos/mop_zeroshot_simclr_checkpoint.pt"
MOP_CKPT_DIR="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_chronos"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_gift_supervised"

echo "============================================================"
echo "MoMS GIFT-Eval — SimCLR-Full SUPERVISED (100% data)"
echo "Goal: upper bound of model capacity on GIFT-Eval"
echo "Encoder: frozen SimCLR-Full"
echo "Fine-tune: 100% GIFT train split, 50 epochs"
echo "============================================================"

time python3 "${SRC}/experiments/mop_full_gift.py" \
  --encoder_name        "${ENCODER}" \
  --checkpoint_dir      "${CKPT_DIR}" \
  --mop_checkpoint_dir  "${MOP_CKPT_DIR}" \
  --stage1_checkpoint   "${STAGE1_CKPT}" \
  --config              "${CONFIG}" \
  --results_dir         "${RESULTS}" \
  --few_shot_fraction   1.0 \
  --finetune_epochs     50 \
  --batch_size          64 \
  --lr                  5e-4 \
  --hidden_dim          512 \
  --num_prompts         16 \
  --context_length      336 \
  --num_workers         0

EXIT_CODE=$?
echo ""
echo "GIFT supervised done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
