#!/bin/bash
#SBATCH --comment=clip_full_probe
#SBATCH --time=720
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clip_full/probe_%j.out
#SBATCH --error=logs/clip_full/probe_%j.err
#SBATCH --job-name=clip_full_probe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/clip_full

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/clip_full
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_clip_full_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No CLIP full checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing CLIP FULL checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_clip_full.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --batch_size 16 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/clip_full \
    --scaler_type standard \
    --seq_len 336 \
    --seed 42
