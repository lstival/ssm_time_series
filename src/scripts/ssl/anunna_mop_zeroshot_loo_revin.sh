#!/bin/bash
#SBATCH --comment=mop_zs_loo_revin
#SBATCH --time=720
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/mop_tuning/mop_zs_loo_revin_%j.out
#SBATCH --error=logs/mop_tuning/mop_zs_loo_revin_%j.err
#SBATCH --job-name=mop_loo_rv
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
echo "Exp B: MoP LOO Zero-Shot — norm_mode=revin (comparable to MoMS)"
echo "============================================================"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/mop_tuning_icml_loo.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_simclr_bimodal_nano.yaml \
  --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/mop_tuning \
  --epochs 100 \
  --batch_size 64 \
  --num_prompts 8 \
  --hidden_dim 512 \
  --context_length 336 \
  --batches_per_epoch 1000 \
  --norm_mode revin \
  --output_suffix _revin

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
