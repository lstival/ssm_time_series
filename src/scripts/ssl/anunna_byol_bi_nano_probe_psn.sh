#!/bin/bash
#SBATCH --comment=byol_bi_nano_probe_psn
#SBATCH --time=60
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/byol_bimodal_nano/probe_psn_%j.out
#SBATCH --error=logs/byol_bimodal_nano/probe_psn_%j.err
#SBATCH --job-name=byol_bi_n_psn
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/byol_bimodal_nano

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/byol_bimodal_nano
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_byol_bimodal_nano_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No byol_bi nano checkpoint found in ${CHECKPOINTS}"; exit 1
fi
echo "Probing [per-series-norm] byol_bi nano: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_byol_bimodal_nano.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --datasets exchange_rate.csv \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/probe_byol_bimodal_nano_psn \
    --scaler_type standard \
    --seq_len 336 \
    --per_series_norm \
    --no_comet \
    --seed 42
