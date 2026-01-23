
import torch
from transformers import AutoModel, AutoConfig
import os
import sys
import traceback

# Add local src to path 
sys.path.append(os.path.abspath("src"))

local_dirs = [
    r"hf_export\cm-mamba-tiny",
    r"hf_export\cm-mamba-mini"
]

for local_dir in local_dirs:
    print(f"\n{'='*20} Testing local {local_dir} {'='*20}")
    if not os.path.exists(local_dir):
        print(f"Directory {local_dir} not found. Skipping.")
        continue
    try:
        print(f"Loading model from {local_dir}...")
        # Since the auto_map is broken (dots issue), AutoModel.from_pretrained will fail even locally 
        # IF it tries to use the dynamic module loader.
        model = AutoModel.from_pretrained(
            local_dir, 
            trust_remote_code=True
        )
        print(f"Successfully loaded {local_dir}")
        print(f"Model type: {type(model)}")
    except Exception as e:
        print(f"Error loading {local_dir}:")
        traceback.print_exc()
