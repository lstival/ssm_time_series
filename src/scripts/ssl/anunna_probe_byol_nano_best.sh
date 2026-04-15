#!/bin/bash
#SBATCH --comment=probe_byol_nano_best
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/probe_lotsa/probe_byol_nano_best_%j.out
#SBATCH --error=logs/probe_lotsa/probe_byol_nano_best_%j.err
#SBATCH --job-name=probe_byol_nano_best
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/probe_lotsa

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/byol_nano_best
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_byol_nano_best_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No BYOL nano best checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing BYOL nano best checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_byol_nano_best.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/probe_byol_nano_best \
    --scaler_type standard \
    --seed 42
