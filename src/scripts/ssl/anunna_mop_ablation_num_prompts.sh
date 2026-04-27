#!/bin/bash
#SBATCH --comment=mop_ablation_num_prompts
#SBATCH --time=720
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mop_tuning/mop_ablation_num_prompts_%j.out
#SBATCH --error=logs/mop_tuning/mop_ablation_num_prompts_%j.err
#SBATCH --job-name=mop_nprmt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/mop_tuning
mkdir -p results/mop_ablation_num_prompts

# HuggingFace cache — same as SimCLR-Full training
export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# SimCLR-Full checkpoint trained on LOTSA+Chronos combined corpus
CHECKPOINT_DIR="checkpoints/simclr_full/ts_simclr_full_lotsa_20260414_002946"

echo "============================================================"
echo "MoP Ablation: num_prompts in [2, 4, 8, 16, 32]"
echo "Encoder:      SimCLR-Full (LOTSA+Chronos, enc_dim=128)"
echo "Train corpus: LOTSA+local combined (same as encoder pre-training)"
echo "Eval:         zero-shot on 7 ICML benchmarks"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_ablation_num_prompts.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_simclr_full.yaml \
  --icml_data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_ablation_num_prompts \
  --epochs 50 \
  --batch_size 64 \
  --lr 1e-3 \
  --hidden_dim 512 \
  --context_length 336 \
  --batches_per_epoch 500 \
  --num_workers 4 \
  --num_prompts_values 2 4 8 16 32

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
