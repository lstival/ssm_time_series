#!/bin/bash
#SBATCH --comment=mop_zeroshot
#SBATCH --time=120
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mop_tuning/mop_zeroshot_%j.out
#SBATCH --error=logs/mop_tuning/mop_zeroshot_%j.err
#SBATCH --job-name=mop_zeroshot
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
MOP_CHECKPOINT="results/mop_tuning/mop_latest.pt"

echo "============================================================"
echo "Starting MoP Zero-Shot Probe"
echo "MoP checkpoint: ${MOP_CHECKPOINT}"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_zeroshot_probe.py \
  --mop_checkpoint "${MOP_CHECKPOINT}" \
  --base_checkpoint_dir "${CHECKPOINT_DIR}" \
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_simclr_bimodal_nano.yaml \
  --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_tuning \
  --batch_size 64 \
  --num_prompts 8 \
  --hidden_dim 512 \
  --context_length 336

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
