#!/bin/bash
#SBATCH --comment=simclr_bi_nano_probe_weather_psn
#SBATCH --time=60
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/simclr_bimodal_nano/probe_weather_psn_%j.out
#SBATCH --error=logs/simclr_bimodal_nano/probe_weather_psn_%j.err
#SBATCH --job-name=simclr_bi_n_wx
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/simclr_bimodal_nano

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
CHECKPOINTS=/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/simclr_bimodal_nano
CHECKPOINT_DIR=$(ls -td ${CHECKPOINTS}/ts_simclr_bimodal_nano_lotsa_* 2>/dev/null | head -1)

if [ -z "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: No simclr_bi nano checkpoint found in ${CHECKPOINTS}"; exit 1
fi
echo "Probing [weather per-series-norm] simclr_bi nano: ${CHECKPOINT_DIR}"

time python3 "${SRC}/experiments/probe_lotsa_checkpoint.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --config "${SRC}/configs/lotsa_simclr_bimodal_nano.yaml" \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets \
    --datasets weather.csv \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/probe_simclr_bimodal_nano_weather_psn \
    --scaler_type standard \
    --seq_len 336 \
    --per_series_norm \
    --no_comet \
    --seed 42
