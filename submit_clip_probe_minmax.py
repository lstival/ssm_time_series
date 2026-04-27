#!/usr/bin/env python3
"""
Submit CLIP Linear Probes with MinMax Scaling.
This allows a fair comparison with MoP results which also use MinMax.
"""

import os
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_BASE = ROOT_DIR / "results" / "clip_minmax_comparison"
# Based on the logs from the previous successful run
CHECKPOINT_DIR = "/lustre/nobackup/WUR/AIN/stiva001/ssm_time_series/checkpoints/clip_full/ts_clip_full_lotsa_20260415_012856"
# Using the standard clip_full config
CONFIG = SRC_DIR / "configs" / "lotsa_clip_full.yaml"

EXPERIMENTS = [
    {"name": "clip_minmax_full", "args": "--few_shot_fraction 1.0"},
    {"name": "clip_minmax_5pct", "args": "--few_shot_fraction 0.05"},
]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --err={log_dir}/{job_name}.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export PYTHONPATH=$PYTHONPATH:{src_dir}

python3 {src_dir}/experiments/probe_lotsa_checkpoint.py \\
    --checkpoint_dir {checkpoint_dir} \\
    --config {config} \\
    --results_dir {out_dir} \\
    --scaler_type minmax \\
    --seq_len 512 \\
    --probe_epochs 20 \\
    {extra_args}
"""

def submit():
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT_DIR / "logs" / "clip_minmax"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    for exp in EXPERIMENTS:
        name = exp["name"]
        extra_args = exp["args"]
        out_dir = RESULTS_BASE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = name
        sbatch_content = SBATCH_TEMPLATE.format(
            job_name=job_name,
            log_dir=log_dir,
            src_dir=SRC_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
            config=CONFIG,
            out_dir=out_dir,
            extra_args=extra_args
        )
        
        sbatch_file = out_dir / "job.sh"
        with open(sbatch_file, "w") as f:
            f.write(sbatch_content)
            
        print(f"Submitting job: {job_name}")
        subprocess.run(["sbatch", str(sbatch_file)])

if __name__ == "__main__":
    submit()
