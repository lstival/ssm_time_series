"""Simple runner script for multi-dataset evaluation."""

import subprocess
import sys
from pathlib import Path


def run_multi_dataset_evaluation(
    checkpoint_path: str,
    model_name: str = "MambaEncoder",
    split: str = "test",
    base_data_dir: str = r"C:\WUR\ssm_time_series",
    output_dir: str = r"C:\WUR\ssm_time_series\eval_runs"
):
    """Run multi-dataset evaluation with given parameters."""
    
    # Build command
    cmd = [
        sys.executable, "-m", "evaluation.multi_dataset_evaluation",
        "--config", r"C:\WUR\ssm_time_series\src\configs\multi_dataset_evaluation.yaml",
        "--checkpoint_path", checkpoint_path,
        "--model_name", model_name,
        "--split", split,
        "--base_data_dir", base_data_dir,
        "--output_dir", output_dir
    ]
    
    print("Running multi-dataset evaluation...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the evaluation
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=base_data_dir)
        if result.returncode == 0:
            print("Evaluation completed successfully!")
        else:
            print(f"Evaluation failed with return code: {result.returncode}")
    except Exception as e:
        print(f"Error running evaluation: {e}")


if __name__ == "__main__":
    # Example configurations for different checkpoints
    checkpoints_to_evaluate = [
        {
            "checkpoint_path": r"C:\WUR\ssm_time_series\checkpoints\forecast_encoder_20251014_2306\best.pt",
            "model_name": "MambaEncoder_v1",
            "split": "test"
        },
        # Add more checkpoints here as needed
        # {
        #     "checkpoint_path": r"C:\WUR\ssm_time_series\checkpoints\forecast_encoder_20251014_2306_v1\best.pt", 
        #     "model_name": "MambaEncoder_v2",
        #     "split": "test"
        # }
    ]
    
    for config in checkpoints_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating: {config['model_name']}")
        print(f"{'='*60}")
        
        run_multi_dataset_evaluation(**config)
        
        print(f"\nCompleted evaluation for {config['model_name']}")
        print("-" * 60)