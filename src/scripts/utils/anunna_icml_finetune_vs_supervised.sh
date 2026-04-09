#!/bin/bash
#SBATCH --comment=icml_finetune_vs_supervised
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icml_compare/icml_finetune_vs_supervised_%j.out
#SBATCH --error=logs/icml_compare/icml_finetune_vs_supervised_%j.err
#SBATCH --job-name=icml_compare
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/icml_compare
mkdir -p results/icml_finetune_vs_supervised

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

_BASE="/home/WUR/stiva001/WUR/ssm_time_series/checkpoints/lotsa_ablation_best"
CHECKPOINT_DIR=$(ls -dt "${_BASE}"/ts_encoder_* 2>/dev/null | head -1)
if [ -z "${CHECKPOINT_DIR}" ]; then
  echo "ERROR: no checkpoint subdir found under ${_BASE}" >&2
  exit 1
fi

echo "============================================================"
echo "ICML Fine-tune vs Supervised Comparison"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Run 1: finetune (pretrained encoders + ICML training)"
echo "Run 2: supervised (random encoders + ICML training)"
echo ""

COMMON_ARGS=(
  --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_ablation_best.yaml
  --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets
  --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/icml_finetune_vs_supervised
  --epochs 50
  --batch_size 128
  --lr 1e-4
  --encoder_lr_scale 0.1
  --mlp_hidden_dim 512
  --context_length 96
  --horizons 96 192 336 720
  --scaler_type standard
)

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/icml_finetune_vs_supervised.py \
  --mode finetune \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  "${COMMON_ARGS[@]}"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/icml_finetune_vs_supervised.py \
  --mode supervised \
  "${COMMON_ARGS[@]}"

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
