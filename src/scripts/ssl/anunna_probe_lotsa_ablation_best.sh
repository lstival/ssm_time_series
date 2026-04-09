#!/bin/bash
#SBATCH --comment=probe_lotsa_ablation_best
#SBATCH --time=1440
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/probe_lotsa/probe_lotsa_ablation_best_%j.out
#SBATCH --error=logs/probe_lotsa/probe_lotsa_ablation_best_%j.err
#SBATCH --job-name=probe_ablation_best
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --dependency=afterok:${ENCODER_JOB_ID}

mkdir -p logs/probe_lotsa
mkdir -p results/probe_lotsa_ablation_best

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

# Use the ablation-best checkpoint directory — auto-detect latest timestamped subdir
_BASE="/home/WUR/stiva001/WUR/ssm_time_series/checkpoints/lotsa_ablation_best"
CHECKPOINT_DIR=$(ls -dt "${_BASE}"/ts_encoder_* 2>/dev/null | head -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoint subdir found under ${_BASE}" >&2
    exit 1
fi

echo "Evaluating ablation-best checkpoint: $CHECKPOINT_DIR"
echo "This job depends on the encoder training job completing successfully."
echo ""

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/probe_lotsa_checkpoint.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --config /home/WUR/stiva001/WUR/ssm_time_series/src/configs/lotsa_ablation_best.yaml \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --scaler_type standard \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/probe_lotsa_ablation_best \
    --datasets ETTm1.csv ETTm2.csv ETTh1.csv ETTh2.csv weather.csv traffic.csv electricity.csv exchange_rate.csv solar_AL.txt

echo ""
echo "Linear probe evaluation complete!"
