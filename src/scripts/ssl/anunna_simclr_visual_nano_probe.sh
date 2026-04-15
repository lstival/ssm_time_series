#!/bin/bash
#SBATCH --comment=probe_simclr_visual_nano
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr_visual_nano/probe_%j.out
#SBATCH --error=logs/simclr_visual_nano/probe_%j.err
#SBATCH --job-name=probe_sv_nano
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/simclr_visual_nano

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/simclr_visual_nano
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_simclr_visual_nano_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No SimCLR visual nano checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing SimCLR visual nano (upper_tri RP) checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_simclr_visual_nano.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/simclr_visual_nano \
    --scaler_type standard \
    --seed 42
