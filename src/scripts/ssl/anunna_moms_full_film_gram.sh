#!/bin/bash
#SBATCH --comment=moms_film_gram
#SBATCH --time=720
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/moms_full/film_gram_%j.out
#SBATCH --error=logs/moms_full/film_gram_%j.err
#SBATCH --job-name=moms_film_gram
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/moms_full results/moms_full_pipeline

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

ENCODER="gram"
CKPT_DIR="checkpoints/gram_full/ts_gram_full_lotsa_20260412_121127"
CONFIG="/home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_gram_full.yaml"
RESULTS="/home/WUR/stiva001/WUR/ssm_time_series/results/moms_full_pipeline"

echo "============================================================"
echo "MoMS FiLM Pipeline — GRAM-Full (Opção C)"
echo "Stage 1: Zero-Shot with fusion_mode=film"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_full_zeroshot.py \
  --encoder_name       "${ENCODER}" \
  --checkpoint_dir     "${CKPT_DIR}" \
  --config             "${CONFIG}" \
  --icml_data_dir      /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir        "${RESULTS}" \
  --epochs             50 \
  --batch_size         64 \
  --lr                 1e-3 \
  --hidden_dim         512 \
  --num_prompts        16 \
  --context_length     336 \
  --batches_per_epoch  500 \
  --num_workers        4 \
  --fusion_mode        film \
  --output_suffix      _film

STAGE1_CKPT="${RESULTS}/mop_zeroshot_${ENCODER}_film_checkpoint.pt"

echo ""
echo "============================================================"
echo "Stage 2: Few-Shot with fusion_mode=film"
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
echo "MoMS FiLM GRAM done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
