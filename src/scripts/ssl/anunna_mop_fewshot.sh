#!/bin/bash
#SBATCH --comment=mop_fewshot
#SBATCH --time=1200
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mop_tuning/mop_fewshot_%j.out
#SBATCH --error=logs/mop_tuning/mop_fewshot_%j.err
#SBATCH --job-name=mop_fewshot
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/mop_tuning
mkdir -p results/mop_tuning

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

CHECKPOINT_DIR="checkpoints/simclr_bimodal_nano/ts_simclr_bimodal_nano_lotsa_20260409_182831"

echo "============================================================"
echo "Starting MoP Few-Shot Probe (5%) — fresh MoP per dataset"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_fewshot_probe.py \
  --base_checkpoint_dir "${CHECKPOINT_DIR}" \
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_simclr_bimodal_nano.yaml \
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_tuning \
  --few_shot_fraction 0.05 \
  --finetune_epochs 50 \
  --batch_size 64 \
  --num_prompts 8 \
  --hidden_dim 512 \
  --context_length 336

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
