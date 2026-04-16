#!/bin/bash
#SBATCH --comment=byol_bimodal_full_probe
#SBATCH --time=480
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/byol_bimodal_full/probe_%j.out
#SBATCH --error=logs/byol_bimodal_full/probe_%j.err
#SBATCH --job-name=byol_bimodal_full_probe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/byol_bimodal_full

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/byol_bimodal_full
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_byol_bimodal_full_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No BYOL bimodal full checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing BYOL bimodal FULL checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_byol_bimodal_full.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --batch_size 16 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/byol_bimodal_full \
    --scaler_type standard \
    --seq_len 336 \
    --no_comet \
    --seed 42
