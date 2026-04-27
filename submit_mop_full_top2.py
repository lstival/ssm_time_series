#!/usr/bin/env python3
"""
Submit Full Training for Top 2 MoP Variations.
Top 2: revin_mlp and revin.
"""

import os
import subprocess
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
RESULTS_BASE = ROOT_DIR / "results" / "mop_full_v1"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "clip_nano" / "ts_clip_nano_lotsa_20260409_181021"
CONFIG = SRC_DIR / "configs" / "lotsa_clip_nano.yaml"

# Top 2 Variations from Grid Search
EXPERIMENTS = [
    {"name": "top1_revin_mlp", "args": "--norm_mode revin --head_type mlp --use_ln_head"},
    {"name": "top2_revin", "args": "--norm_mode revin --head_type linear"},
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
#SBATCH --time=12:00:00

module load GPU
source /home/WUR/stiva001/WUR/timeseries/bin/activate

export PYTHONPATH=$PYTHONPATH:{src_dir}

# 1. Full Training on Source (LOTSA)
python3 {src_dir}/experiments/mop_train_flex.py \\
    --checkpoint_dir {checkpoint_dir} \\
    --config {config} \\
    --results_dir {out_dir} \\
    --mode source \\
    --epochs 50 \\
    --batches_per_epoch 2000 \\
    {extra_args}

# 2. Zero-Shot Evaluation (Full Scale)
python3 {src_dir}/experiments/mop_eval_flex.py \\
    --mop_checkpoint {out_dir}/mop_flex.pt \\
    --checkpoint_dir {checkpoint_dir} \\
    --config {config} \\
    --results_dir {out_dir}/eval_zeroshot_full
"""

def submit():
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT_DIR / "logs" / "mop_full"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    for exp in EXPERIMENTS:
        name = exp["name"]
        extra_args = exp["args"]
        out_dir = RESULTS_BASE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = f"mop_full_{name}"
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
            
        print(f"Submitting full job: {job_name}")
        subprocess.run(["sbatch", str(sbatch_file)])

if __name__ == "__main__":
    submit()
