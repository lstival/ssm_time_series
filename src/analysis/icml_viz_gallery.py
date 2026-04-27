import os
import subprocess
import sys
from pathlib import Path

# Paths relative to project root
DATA_PATHS = {
    "ETTh1": "ICML_datasets/ETT-small/ETTh1.csv",
    "ETTh2": "ICML_datasets/ETT-small/ETTh2.csv",
    "ETTm1": "ICML_datasets/ETT-small/ETTm1.csv",
    "ETTm2": "ICML_datasets/ETT-small/ETTm2.csv",
    "Weather": "ICML_datasets/weather/weather.csv",
    "Traffic": "ICML_datasets/traffic/traffic.csv",
    "Electricity": "ICML_datasets/electricity/electricity.csv",
}

# Characteristic channels (Traffic and Electricity have many, others less)
# We'll stick to 0 for consistency unless they're known to be flat.
CHANNELS = {
    "ETTh1": 0, "ETTh2": 0, "ETTm1": 0, "ETTm2": 0,
    "Weather": 0, "Traffic": 0, "Electricity": 0
}

SRC_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = SRC_DIR.parent
CHECKPOINT = "checkpoints/byol_bimodal_full/ts_byol_bimodal_full_lotsa_20260414_171343"

def main():
    gallery_dir = ROOT_DIR / "results" / "icml_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    # Add src to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + ":" + env.get("PYTHONPATH", "")
    
    viz_script = SRC_DIR / "analysis" / "mamba_attn_viz.py"

    for name, rel_path in DATA_PATHS.items():
        data_file = ROOT_DIR / rel_path
        if not data_file.exists():
            print(f"Skipping {name}: {data_file} not found")
            continue
            
        print(f"\n>>> Generating viz for {name} ({rel_path})...")
        out_file = gallery_dir / f"viz_{name.lower()}.png"
        
        cmd = [
            "python3", str(viz_script),
            "--checkpoint-dir", str(ROOT_DIR / CHECKPOINT),
            "--etth1", str(data_file),
            "--channel", str(CHANNELS[name]),
            "--output", str(out_file),
            "--device", "auto"
        ]
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ Validated and saved to {out_file.name}")
            else:
                print(f"  ✗ Failed for {name}")
                print(result.stderr)
        except Exception as e:
            print(f"  ✗ Error running for {name}: {e}")

if __name__ == "__main__":
    main()
