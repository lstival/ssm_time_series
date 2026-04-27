#!/bin/bash
#SBATCH --comment=simclr_mv_delay_train
#SBATCH --time=720
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr_mv/delay_%j.out
#SBATCH --error=logs/simclr_mv/delay_%j.err
#SBATCH --job-name=mv_delay
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
echo "Ablation MV-B: SimCLR Full — Delay Embedding RP (Takens)"
echo "rp_mv_strategy=delay_embed | 24h budget | LOTSA+Chronos"
echo "============================================================"

time python3 "${SRC}/simclr_bimodal_training.py" \
    --config "${SRC}/configs/lotsa_simclr_full_mv_delay.yaml"

EXIT_CODE=$?
echo "Training done. Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
