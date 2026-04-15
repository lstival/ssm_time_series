#!/bin/bash
#SBATCH --job-name=Unimodal_evaluted_CLIP
#SBATCH --comment=Unimodal_evaluted_CLIP
#SBATCH --time=60
#SBATCH --mem=4000
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/clip_full/evaluate_notify_%j.out
#SBATCH --error=logs/clip_full/evaluate_notify_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leandroteso@gmail.com
#SBATCH --partition=main

mkdir -p logs/clip_full

source /home/WUR/stiva001/WUR/timeseries/bin/activate

SRC=/home/WUR/stiva001/WUR/ssm_time_series/src
RESULTS=/home/WUR/stiva001/WUR/ssm_time_series/results

echo "=== Generating updated unimodal comparison table ==="
echo "Using CLIP full multimodal results from: ${RESULTS}/clip_full/probe_results_full.csv"

# Check CSV exists
if [ ! -f "${RESULTS}/clip_full/probe_results_full.csv" ]; then
    echo "ERROR: CLIP full probe results not found at ${RESULTS}/clip_full/probe_results_full.csv"
    exit 1
fi

# Generate updated table using CLIP full (full-size multimodal) instead of micro
python3 "${SRC}/scripts/plots/generate_unimodal_comparison_table.py" \
    --temporal_csv    "${RESULTS}/clip_temporal/probe_lotsa_results.csv" \
    --visual_csv      "${RESULTS}/clip_visual/probe_results_full.csv" \
    --multimodal_csv  "${RESULTS}/clip_full/probe_results_full.csv" \
    --out             "${RESULTS}/table_unimodal_comparison.tex"

echo "=== Table generated ==="
cat "${RESULTS}/table_unimodal_comparison.tex"
