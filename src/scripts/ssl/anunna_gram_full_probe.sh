#!/bin/bash
#SBATCH --comment=gram_full_probe
#SBATCH --time=480
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gram_full/probe_%j.out
#SBATCH --error=logs/gram_full/probe_%j.err
#SBATCH --job-name=gram_full_probe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/gram_full

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/gram_full
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_gram_full_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No GRAM full checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing GRAM FULL checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_gram_full.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/gram_full \
    --scaler_type standard \
    --seed 42
