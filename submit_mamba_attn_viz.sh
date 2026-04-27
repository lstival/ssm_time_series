#!/bin/bash
#SBATCH --job-name=mamba_attn_viz
#SBATCH --output=/home/WUR/stiva001/WUR/ssm_time_series/logs/mamba_attn_viz_%j.out
#SBATCH --err=/home/WUR/stiva001/WUR/ssm_time_series/logs/mamba_attn_viz_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC="/home/WUR/stiva001/WUR/ssm_time_series/src"
export PYTHONPATH=$PYTHONPATH:${SRC}

python3 "${SRC}/analysis/mamba_attn_viz.py" \
    --checkpoint-dir /home/WUR/stiva001/WUR/ssm_time_series/checkpoints/byol_bimodal_full/ts_byol_bimodal_full_lotsa_20260414_171343 \
    --etth1 /home/WUR/stiva001/WUR/ssm_time_series/ICML_datasets/ETT-small/ETTh1.csv \
    --context-len 336 \
    --start-idx 0 \
    --channel 6 \
    --output /home/WUR/stiva001/WUR/ssm_time_series/results/mamba_attn_viz.png \
    --device auto

echo "Done. Plots at:"
echo "  /home/WUR/stiva001/WUR/ssm_time_series/results/mamba_attn_viz.png"
echo "  /home/WUR/stiva001/WUR/ssm_time_series/results/mamba_attn_viz_rp_panel.png"
