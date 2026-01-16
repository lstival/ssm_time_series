"""Experiment runner for EXP-2: Recurrence Plot Information Leakage Test."""

import os
import subprocess
import sys
from pathlib import Path

def run_experiment(rp_mode, experiment_name):
    print(f"\n>>> Starting Experiment: {experiment_name} (RP Mode: {rp_mode})")
    
    cmd = [
        "conda", "run", "-n", "timeseries", "python",
        "src/contrastive_training.py",
        "--rp-mode", rp_mode,
        "--epochs", "2",  # Reduced for sanity check, adjust as needed
        "--batch-size", "64",
        "--checkpoint-dir", f"checkpoints/exp2/{rp_mode}"
    ]
    
    # Ensure CWD is root
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    if process.returncode != 0:
        print(f"!!! Experiment {experiment_name} failed with return code {process.returncode}")
    else:
        print(f">>> Completed Experiment: {experiment_name}")

def main():
    # Variants to run
    variants = [
        ("correct", "Correct RP (Baseline)"),
        ("shuffled", "Shuffled RP (Negative Control)"),
        ("random", "Random RP (Noise Control)")
    ]
    
    for rp_mode, name in variants:
        run_experiment(rp_mode, name)
    
    print("\n\n" + "="*50)
    print("EXP-2: Recurrence Plot Information Leakage Test Completed.")
    print("Please check the logs and checkpoints in checkpoints/exp2/")
    print("="*50)

if __name__ == "__main__":
    main()
