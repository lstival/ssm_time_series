#!/bin/bash
#SBATCH --comment=icml_simclr_micro_probe
#SBATCH --time=360
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icml_simclr_micro/probe_%j.out
#SBATCH --error=logs/icml_simclr_micro/probe_%j.err
#SBATCH --job-name=icml_simclr_mp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/icml_simclr_micro

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/icml_simclr_micro
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_simclr_micro_icml_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No SimCLR micro ICML checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing SimCLR micro ICML checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/icml_simclr_micro.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/icml_simclr_micro \
    --scaler_type standard \
    --seed 42
