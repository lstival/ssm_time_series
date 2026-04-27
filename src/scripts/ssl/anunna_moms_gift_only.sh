#!/bin/bash
#SBATCH --comment=moms_gift_only
#SBATCH --time=240
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/gift_only_%j.out
#SBATCH --error=logs/moms_full/gift_only_%j.err
#SBATCH --job-name=moms_gift
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
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_chronos"

run_gift() {
    local ENCODER=$1
    local CKPT_DIR=$2
    local CONFIG=$3
    local STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}_checkpoint.pt"

    echo ""
    echo "============================================================"
    echo "GIFT-Eval Stage 3 — ${ENCODER^^} (z-score fix)"
    echo "============================================================"
    time python3 "${SRC}/experiments/mop_full_gift.py" \
      --encoder_name        "${ENCODER}" \
      --checkpoint_dir      "${CKPT_DIR}" \
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
      --num_workers         0
}

run_gift "byol"   "checkpoints/byol_bimodal_full/ts_byol_bimodal_full_lotsa_20260414_171343" \
                  "${SRC}/configs/lotsa_byol_bimodal_full.yaml"

run_gift "simclr" "checkpoints/simclr_full/ts_simclr_full_lotsa_20260414_002946" \
                  "${SRC}/configs/lotsa_simclr_full.yaml"

run_gift "gram"   "checkpoints/gram_full/ts_gram_full_lotsa_20260412_121127" \
                  "${SRC}/configs/lotsa_gram_full.yaml"

run_gift "clip"   "checkpoints/clip_full/ts_clip_full_lotsa_20260415_012856" \
                  "${SRC}/configs/lotsa_clip_full.yaml"

EXIT_CODE=$?
echo ""
echo "GIFT-Eval re-run done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
