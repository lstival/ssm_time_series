#!/bin/bash
#SBATCH --comment=icml_vl_jepa_micro_train
#SBATCH --time=720
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/icml_vl_jepa_micro/train_%j.out
#SBATCH --error=logs/icml_vl_jepa_micro/train_%j.err
#SBATCH --job-name=icml_jepa_m
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

mkdir -p logs/icml_vl_jepa_micro

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Training VL-JEPA MICRO on ICML datasets (supervised-domain baseline)"
echo "  model_dim=64, depth=6, emb=32, ctx=512, epochs=50, ema_tau=0.996"
echo "  datasets: ett_h1/h2, ett_m1/m2, electricity_hourly, traffic_hourly, exchange_rate"

time python3 "${SRC}/vl_jepa_training.py" \
    --config "${SRC}/configs/icml_vl_jepa_micro.yaml"
