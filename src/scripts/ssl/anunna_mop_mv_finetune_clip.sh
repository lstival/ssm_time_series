#!/bin/bash
#SBATCH --comment=mop_mv_finetune_clip
#SBATCH --time=480
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/mv_finetune_clip_%j.out
#SBATCH --error=logs/moms_full/mv_finetune_clip_%j.err
#SBATCH --job-name=mv_ft_clip
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full results/mop_mv_finetune

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

ENCODER="clip"
CKPT_DIR="checkpoints/clip_full/ts_clip_full_lotsa_20260415_012856"
CONFIG="/home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_clip_full.yaml"
STAGE1_CKPT="results/moms_full_pipeline/mop_zeroshot_clip_checkpoint.pt"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/mop_mv_finetune"

echo "============================================================"
echo "MoMS MV Fine-Tune Test — CLIP-Full"
echo "Datasets: weather, traffic, electricity"
echo "Fractions: 0% (baseline), 5%, 20%, 50%, 100%"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_mv_finetune_test.py \
  --encoder_name     "${ENCODER}" \
  --checkpoint_dir   "${CKPT_DIR}" \
  --mop_checkpoint   "${STAGE1_CKPT}" \
  --config           "${CONFIG}" \
  --icml_data_dir    /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir      "${RESULTS}" \
  --fractions        0.05 0.20 0.50 1.00 \
  --finetune_epochs  20 \
  --batch_size       64 \
  --lr               5e-4 \
  --hidden_dim       512 \
  --num_prompts      16 \
  --context_length   336 \
  --num_workers      0

EXIT_CODE=$?
echo "Done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
