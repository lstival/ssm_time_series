#!/bin/bash
#SBATCH --comment=download_lotsa_data
#SBATCH --time=3-00:00:00
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/download_lotsa/download_lotsa_%j.out
#SBATCH --error=logs/download_lotsa/download_lotsa_%j.err
#SBATCH --job-name=download_lotsa_data
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl
#SBATCH --partition=main

mkdir -p logs/download_lotsa

# Lustre no-backup storage
LOTSA_CACHE="/lustre/nobackup/WUR/AIN/stiva001/hf_datasets"

source /home/WUR/stiva001/WUR/timeseries/bin/activate

mkdir -p "${LOTSA_CACHE}"

echo "Starting download of Salesforce/lotsa_data"
echo "Cache directory: ${LOTSA_CACHE}"
echo "Start time: $(date)"

time python3 /home/WUR/stiva001/WUR/ssm_time_series/src/download_lotsa_data.py \
    --cache_dir "${LOTSA_CACHE}"

echo "End time: $(date)"

# How to submit:
# sbatch src/scripts/download_lotsa_data.sh
