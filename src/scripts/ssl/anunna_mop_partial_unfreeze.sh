#!/bin/bash
#SBATCH --comment=mop_partial_unfreeze
#SBATCH --time=480
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mop_tuning/mop_partial_unfreeze_%j.out
#SBATCH --error=logs/mop_tuning/mop_partial_unfreeze_%j.err
#SBATCH --job-name=mop_unfreeze
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
echo "MoP Partial Unfreeze: 10ep warmup + 40ep last-2-blocks"
echo "Encoder: SimCLR Nano Bimodal"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_partial_unfreeze.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_simclr_bimodal_nano.yaml \
  --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_tuning \
  --batch_size 64 \
  --lr_base 1e-3 \
  --lr_encoder_mult 0.1 \
  --num_last_blocks 2 \
  --num_prompts 8 \
  --hidden_dim 512 \
  --context_length 336 \
  --batches_per_epoch 500

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
