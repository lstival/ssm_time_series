#!/bin/bash
#SBATCH --comment=plot_icml_series_examples
#SBATCH --time=120
#SBATCH --mem=16000
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/plots/icml_series_examples_%j.out
#SBATCH --error=logs/plots/icml_series_examples_%j.err
#SBATCH --job-name=plot_icml_examples
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leandro.stival@wur.nl

set -euo pipefail

REPO=/home/WUR/stiva001/WUR/ssm_time_series
mkdir -p "${REPO}/logs/plots"

module load GPU || true
source /home/WUR/stiva001/WUR/timeseries/bin/activate

cd "${REPO}"

python3 src/analysis/plot_icml_series_examples.py
