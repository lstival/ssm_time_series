#!/bin/bash
#SBATCH --comment=benchmark_rp_overhead
#SBATCH --time=60
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke/rp_benchmark_%j.out
#SBATCH --error=logs/smoke/rp_benchmark_%j.err
#SBATCH --job-name=rp_bench
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

mkdir -p logs/smoke

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
REPO=/home/WUR/stiva001/WUR/ssm_time_series

echo "Start: $(date)"

time python3 "${SRC}/experiments/smoke_test/benchmark_rp_overhead.py" \
    --config   "${SRC}/experiments/smoke_test/smoke_config.yaml" \
    --data_dir "${REPO}/ICML_datasets/ETT-small" \
    --dataset_type icml \
    --n_batches 100 \
    --seed 42

echo "End: $(date)"
