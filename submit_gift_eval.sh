#!/bin/bash
#SBATCH --job-name=gift_eval_zeroshot
#SBATCH --output=/home/WUR/stiva001/WUR/ssm_time_series/logs/gift_eval/gift_eval_%j.out
#SBATCH --err=/home/WUR/stiva001/WUR/ssm_time_series/logs/gift_eval/gift_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/WUR/stiva001/WUR/ssm_time_series/src

# Run the evaluation script
python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/eval_gift_mop_zeroshot.py \
    --results_dir /home/WUR/stiva001/WUR/ssm_time_series/results/gift_eval_zeroshot
