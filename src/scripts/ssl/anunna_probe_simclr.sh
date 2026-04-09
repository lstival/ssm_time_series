#!/bin/bash
# SimCLR linear probe — temporal or visual branch.
# Usage: MODE=temporal sbatch anunna_probe_simclr.sh
#        MODE=visual   sbatch anunna_probe_simclr.sh
#SBATCH --comment=probe_simclr
#SBATCH --time=1440
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr/probe_%j.out
#SBATCH --error=logs/simclr/probe_%j.err
#SBATCH --job-name=probe_simclr
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

MODE="${MODE:-temporal}"   # temporal | visual
mkdir -p logs/simclr results/probe_simclr_${MODE}

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

_BASE="/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/simclr_${MODE}"
CHECKPOINT_DIR=$(ls -dt "${_BASE}"/ts_simclr_${MODE}_* 2>/dev/null | head -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No checkpoint found under ${_BASE}" >&2
    exit 1
fi

echo "Evaluating SimCLR ${MODE} checkpoint: $CHECKPOINT_DIR"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --config "${SRC}/configs/lotsa_simclr_${MODE}.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --scaler_type standard \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/probe_simclr_${MODE} \
    --datasets ETTm1.csv ETTm2.csv ETTh1.csv ETTh2.csv weather.csv traffic.csv electricity.csv exchange_rate.csv solar_AL.txt

echo "Linear probe (SimCLR ${MODE}) complete!"
