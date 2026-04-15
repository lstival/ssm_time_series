#!/bin/bash
#SBATCH --comment=simclr_full_probe
#SBATCH --time=720
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr_full/probe_%j.out
#SBATCH --error=logs/simclr_full/probe_%j.err
#SBATCH --job-name=simclr_full_probe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/simclr_full

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/simclr_full
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_simclr_full_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No SimCLR full checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing SimCLR FULL checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_simclr_full.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --batch_size 16 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/simclr_full \
    --scaler_type standard \
    --seq_len 336 \
    --no_comet \
    --seed 42
