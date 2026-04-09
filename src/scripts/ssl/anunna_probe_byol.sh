#!/bin/bash
# BYOL linear probe — temporal or visual branch.
# Usage: MODE=temporal sbatch anunna_probe_byol.sh
#        MODE=visual   sbatch anunna_probe_byol.sh
# Defaults to "temporal" if MODE is unset.
#SBATCH --comment=probe_byol
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/byol/probe_%j.out
#SBATCH --error=logs/byol/probe_%j.err
#SBATCH --job-name=probe_byol
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

MODE="${MODE:-temporal}"   # temporal | visual
mkdir -p logs/byol logs/probe_lotsa

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/byol_${MODE}

CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_byol_${MODE}_lotsa_* 2>/dev/null | head -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "ERROR: No byol_${MODE} checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing BYOL ${MODE} checkpoint: $CHECKPOINT_DIR"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --config "${SRC}/configs/lotsa_byol_${MODE}.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/probe_byol_${MODE} \
    --scaler_type standard \
    --seed 42
