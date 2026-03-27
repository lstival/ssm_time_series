#!/bin/bash
#SBATCH --comment=exp2_rp_leakage
#SBATCH --time=4320
#SBATCH --mem=20000
#SBATCH --cpus-per-task=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#SBATCH --job-name=exp2_rp_leakage
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

time srun python3 /home/WUR/stiva001/WUR/ssm_time_series/src/experiments/exp2_rp_leakage.py

# how to call the job
# sbatch src/run_exp2_rp_leakage.sh
