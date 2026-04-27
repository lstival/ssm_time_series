#!/bin/bash
#SBATCH --job-name=test_attn
#SBATCH --output=/home/WUR/stiva001/WUR/ssm_time_series/logs/test2_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

python scratch/test2.py
