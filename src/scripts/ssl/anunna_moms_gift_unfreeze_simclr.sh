#!/bin/bash
#SBATCH --comment=moms_gift_unfreeze_simclr
#SBATCH --time=480
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/gift_unfreeze_simclr_%j.out
#SBATCH --error=logs/moms_full/gift_unfreeze_simclr_%j.err
#SBATCH --job-name=gift_unfrz_sim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full
mkdir -p results/moms_gift_unfreeze

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
ENCODER="simclr"
CKPT_DIR="checkpoints/simclr_full/ts_simclr_full_lotsa_20260412_121127"
CONFIG="${SRC}/configs/lotsa_simclr_full.yaml"
STAGE1_CKPT="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_chronos/mop_zeroshot_simclr_checkpoint.pt"
MOP_CKPT_DIR="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_chronos"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_gift_unfreeze"

echo "============================================================"
echo "MoMS GIFT-Eval — SimCLR-Full Gradual Encoder Unfreeze"
echo "100% GIFT train data (supervised upper bound)"
echo "Phase 1: 10ep heads+MoP only    lr=5e-4"
echo "Phase 2: 10ep + last block       lr=1e-4"
echo "Phase 3: 30ep + full encoder     lr=5e-5"
echo "============================================================"

time python3 "${SRC}/experiments/mop_gift_unfreeze.py" \
  --encoder_name        "${ENCODER}" \
  --checkpoint_dir      "${CKPT_DIR}" \
  --mop_checkpoint_dir  "${MOP_CKPT_DIR}" \
  --stage1_checkpoint   "${STAGE1_CKPT}" \
  --config              "${CONFIG}" \
  --results_dir         "${RESULTS}" \
  --few_shot_fraction   1.0 \
  --phase1_epochs       10 \
  --phase2_epochs       10 \
  --phase3_epochs       30 \
  --head_lr             5e-4 \
  --enc_lr_last         1e-4 \
  --enc_lr_full         5e-5 \
  --batch_size          64 \
  --hidden_dim          512 \
  --num_prompts         16 \
  --context_length      336 \
  --num_workers         0

EXIT_CODE=$?
echo ""
echo "GIFT unfreeze done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
