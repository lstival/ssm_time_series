#!/bin/bash
#SBATCH --comment=lookback_probe
#SBATCH --time=240
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/lookback/probe_%a_%j.out
#SBATCH --error=logs/lookback/probe_%a_%j.err
#SBATCH --job-name=lookback_probe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'
#SBATCH --array=0-4

mkdir -p logs/lookback

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

LOOKBACKS=(96 192 336 512 720)
CTX=${LOOKBACKS[$SLURM_ARRAY_TASK_ID]}
CONFIG="${SRC}/configs/lotsa_lookback_${CTX}.yaml"
CHECKPOINTS="/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/lookback_${CTX}"
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_lookback_${CTX}_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing lookback=${CTX}  checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${CONFIG}" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/lookback_${CTX} \
    --scaler_type standard \
    --seed 42
