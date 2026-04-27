#!/bin/bash
# Master script to run full comparison experiment and generate table

# Submit unimodal pipeline (Job from earlier)
UNI_JOB=$(sbatch --parsable src/scripts/ssl/anunna_clip_uni_gift_ssl.sh)
echo "Submitted Unimodal Job: $UNI_JOB"

# Submit multimodal pipeline (If not already running, using current job as reference)
# In this environment, we assume Job 66537292 is the current bimodal run.
BI_JOB="66537292"

# Submit reporting job (depends on both)
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=moms_reporting
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --output=logs/reporting_%j.out
#SBATCH --dependency=afterok:${UNI_JOB}:${BI_JOB}

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate
export PYTHONPATH=\$PYTHONPATH:/home/WUR/stiva001/WUR/ssm_time_series/src

python3 /home/WUR/stiva001/WUR/ssm_time_series/src/scripts/reporting/gen_comparison_latex.py
EOT

echo "Submitted Reporting Job dependent on $UNI_JOB and $BI_JOB"
