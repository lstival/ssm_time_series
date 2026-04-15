#!/bin/bash
# CLIP unimodal linear probe — temporal or visual branch.
# Usage: MODE=temporal sbatch anunna_clip_unimodal_probe.sh
#        MODE=visual   sbatch anunna_clip_unimodal_probe.sh
# Defaults to "temporal" if MODE is unset.
#SBATCH --comment=clip_unimodal_probe
#SBATCH --time=480
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/clip_unimodal/probe_%j.out
#SBATCH --error=logs/clip_unimodal/probe_%j.err
#SBATCH --job-name=clip_uni_probe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

MODE="${MODE:-temporal}"   # temporal | visual
mkdir -p logs/clip_unimodal

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/clip_${MODE}
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_clip_${MODE}_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No CLIP ${MODE} checkpoint found in ${CHECKPOINTS}"
    exit 1
fi

echo "Probing CLIP ${MODE} checkpoint: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_clip_${MODE}.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/clip_${MODE} \
    --scaler_type standard \
    --seed 42
