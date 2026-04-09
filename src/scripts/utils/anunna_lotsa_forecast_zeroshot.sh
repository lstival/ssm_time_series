#!/bin/bash
#SBATCH --comment=lotsa_forecast_zeroshot
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/lotsa_zeroshot/lotsa_forecast_zeroshot_%j.out
#SBATCH --error=logs/lotsa_zeroshot/lotsa_forecast_zeroshot_%j.err
#SBATCH --job-name=lotsa_zeroshot
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/lotsa_zeroshot
mkdir -p results/lotsa_zeroshot

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# Auto-detect latest ablation-best checkpoint
_BASE="/home/WUR/stiva001/WUR/ssm_time_series/checkpoints/lotsa_ablation_best"
CHECKPOINT_DIR=$(ls -dt "${_BASE}"/ts_encoder_* 2>/dev/null | head -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoint subdir found under ${_BASE}" >&2
    exit 1
fi

echo "============================================================"
echo "LOTSA Forecast + Zero-shot Evaluation"
echo "============================================================"
echo "Checkpoint : $CHECKPOINT_DIR"
echo "Stage 1    : Train MLP forecasting head on LOTSA (frozen encoders)"
echo "Stage 2    : Zero-shot evaluation on ICML benchmark datasets"
echo "             (no fine-tuning, direct inference on unseen data)"
echo ""

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/lotsa_forecast_zeroshot.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_ablation_best.yaml \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/lotsa_zeroshot \
    --horizons 96 192 336 720 \
    --context_length 96 \
    --mlp_epochs 20 \
    --mlp_hidden_dim 512 \
    --batch_size 256 \
    --lr 1e-3

EXIT_CODE=$?
echo ""
echo "Job completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
