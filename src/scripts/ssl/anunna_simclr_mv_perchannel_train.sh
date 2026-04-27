#!/bin/bash
#SBATCH --comment=simclr_mv_perchannel_train
#SBATCH --time=720
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr_mv/perchannel_%j.out
#SBATCH --error=logs/simclr_mv/perchannel_%j.err
#SBATCH --job-name=mv_perchan
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
set -euo pipefail

mkdir -p logs/simclr_mv

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "============================================================"
echo "Ablation MV-D: SimCLR Full — Per-Channel RP stacking (avg)"
echo "rp_mv_strategy=per_channel | 24h budget | LOTSA+Chronos"
echo "============================================================"

time python3 "${SRC}/simclr_bimodal_training.py" \
    --config "${SRC}/configs/lotsa_simclr_full_mv_perchannel.yaml"

EXIT_CODE=$?
echo "Training done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
