#!/bin/bash
#SBATCH --comment=moms_fewshot_clip_film
#SBATCH --time=600
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/fewshot_clip_film_%j.out
#SBATCH --error=logs/moms_full/fewshot_clip_film_%j.err
#SBATCH --job-name=moms_fs_clip_film
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full
mkdir -p results/moms_full_pipeline

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

ENCODER="clip"
CKPT_DIR="checkpoints/clip_full/ts_clip_full_lotsa_20260415_012856"
CONFIG="/home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_clip_full.yaml"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_pipeline"
STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}_checkpoint.pt"

echo "============================================================"
echo "MoMS Few-Shot FiLM — CLIP-Full (Opção C: no encoder retrain)"
echo "fusion_mode=film | Stage 1 checkpoint: ${STAGE1_CKPT}"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_full_fewshot.py \
  --encoder_name       "${ENCODER}" \
  --checkpoint_dir     "${CKPT_DIR}" \
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
  --fusion_mode        film \
  --output_suffix      _film

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
