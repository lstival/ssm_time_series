#!/bin/bash
#SBATCH --comment=download_lotsa_full
#SBATCH --time=480
#SBATCH --mem=8000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/download_lotsa/download_lotsa_full_%j.out
#SBATCH --error=logs/download_lotsa/download_lotsa_full_%j.err
#SBATCH --job-name=download_lotsa_full
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --partition=main

mkdir -p logs/download_lotsa

# No module needed for CPU job
source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src

# Download full Salesforce/lotsa_data to HF cache
# All 52 subsets: m4, m3, m1, ETT, electricity, traffic, solar, weather, etc.
time python3 "${SRC}/download_lotsa_data.py"

# sbatch src/scripts/download_lotsa_full.sh
