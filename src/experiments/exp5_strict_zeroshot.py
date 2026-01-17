"""
EXP-5: Strict Cross-Domain Zero-Shot Evaluation Runner.
Objective: Validate true zero-shot generalization by removing all overlapping domains from pretraining.
"""

import sys
import os
from pathlib import Path

# Add src to pythonpath
sys.path.append(str(Path(__file__).resolve().parent.parent))

from contrastive_training import main as train_main

def main():
    print("Starting EXP-5: Strict Cross-Domain Zero-Shot Evaluation...")
    
    # Paths
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    config_dir = src_dir / "configs"
    
    # Config files
    cronos_config = config_dir / "exp5_strict_zeroshot_datasets.yaml"
    model_config = config_dir / "mamba_encoder.yaml"
    
    # Checkpoints
    # We want a specific directory for this experiment to avoid overwriting or mixing
    checkpoint_dir = Path("checkpoints/exp5_strict_zeroshot")
    
    # Construct arguments
    # We'll default to the settings in mamba_encoder.yaml but override the dataset config
    # and checkpoint directory.
    
    argv = [
        "--config", str(model_config),
        "--cronos-config", str(cronos_config),
        "--checkpoint-dir", str(checkpoint_dir),
        # "--experiment-name" is not supported by contrastive_training.py's argparse
        # The experiment name in 'mamba_encoder.yaml' ("ts_encoder") will be used for the run directory formatting.

        # The argparse in contrastive_training.py doesn't seem to have --experiment-name.
        # It relies on the config file or creates a run dir based on it.
        # However, resolve_checkpoint_dir uses config.experiment_name.
        # We might need to rely on the config's experiment name 'ts_encoder' but the checkpoint-dir argument
        # helps separate it.
    ]
    
    # Basic check for dry run arg (passed from command line to this script)
    if "--dry-run" in sys.argv:
        print("Dry run mode enabled: 1 epoch")
        argv.extend(["--epochs", "1"])
    
    print(f"Running with args: {argv}")
    
    # Run training
    train_main(argv)

if __name__ == "__main__":
    main()
