#!/bin/bash
# SimCLR single-encoder training — temporal or visual branch.
# Usage: MODE=temporal sbatch anunna_simclr.sh
#        MODE=visual   sbatch anunna_simclr.sh
# Defaults to "temporal" if MODE is unset.
#
# Temporal: MambaEncoder, two augmented temporal views, NT-Xent temp=0.07.
# Visual:   UpperTriDiagRPEncoder, two augmented RP views, NT-Xent temp=0.07.
#SBATCH --comment=simclr_train
#SBATCH --time=2880
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr/train_%j.out
#SBATCH --error=logs/simclr/train_%j.err
#SBATCH --job-name=simclr_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

MODE="${MODE:-temporal}"   # temporal | visual
mkdir -p logs/simclr

export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Starting SimCLR ${MODE} encoder training (100 epochs)..."
echo "Config: lotsa_simclr_${MODE}.yaml"

time python3 "${SRC}/simclr_training.py" \
    --config "${SRC}/configs/lotsa_simclr_${MODE}.yaml" \
    --mode "${MODE}"

EXIT_CODE=$?
echo "SimCLR ${MODE} training finished with exit code: $EXIT_CODE"
exit $EXIT_CODE
