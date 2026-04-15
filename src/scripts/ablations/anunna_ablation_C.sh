#!/bin/bash
#SBATCH --comment=ablation_C_alignment
#SBATCH --time=2880
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ablation_C/train_%j.out
#SBATCH --error=logs/ablation_C/train_%j.err
#SBATCH --job-name=ablation_C
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/ablation_C

export HF_HOME="/lustre/nobackup/WUR/AIN/stiva001/hf_cache"
export HF_DATASETS_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_cache/datasets"
export TMPDIR="/lustre/nobackup/WUR/AIN/stiva001/tmp"
mkdir -p "${HF_DATASETS_CACHE}" "${TMPDIR}"

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

echo "Ablation C — Alignment Strategy (comparable to final models)"
echo "  config:       lotsa_clip_nano.yaml (ctx=336, nano arch)"
echo "  train_epochs: 100"
echo "  probe_epochs: 20"
echo "  seq_len:      336"
echo "  datasets:     8 ICML datasets"

time python3 "${SRC}/experiments/ablation_C_alignment.py" \
    --config "${SRC}/configs/lotsa_clip_nano.yaml" \
    --train_epochs 100 \
    --probe_epochs 20 \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/ablation_C_v2 \
    --data_dir /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets
