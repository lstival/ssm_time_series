
import torch
import os
import sys
import traceback
from pathlib import Path

# Add local src to path 
sys.path.append(os.path.abspath("src"))

from ssm_time_series.hf.forecasting import CM_MambaForecastModel

local_dirs = [
    r"hf_export\cm-mamba-tiny",
    r"hf_export\cm-mamba-mini"
]

for local_dir in local_dirs:
    print(f"\n{'='*20} Testing direct load {local_dir} {'='*20}")
    if not os.path.exists(local_dir):
        print(f"Directory {local_dir} not found. Skipping.")
        continue
    try:
        print(f"Loading model from {local_dir}...")
        model = CM_MambaForecastModel.from_pretrained(local_dir)
        print(f"Successfully loaded {local_dir}")
        print(f"Model type: {type(model)}")
        
        # Test a forward pass
        dummy_input = torch.randn(1, 48, 32) # [B, T, F]
        output = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Error loading {local_dir}:")
        traceback.print_exc()
