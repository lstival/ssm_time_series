
import torch
from transformers import AutoModel, AutoConfig
import os
import sys
import traceback

# Add local src to path to ensure it finds the classes if they are not in the repo
# sys.path.append(os.path.abspath("src"))

repo_id = "lstival/CM-Mamba"
subfolders = ["cm-mamba-tiny", "cm-mamba-mini"]

for subfolder in subfolders:
    print(f"\n{'='*20} Testing {subfolder} {'='*20}")
    try:
        print(f"Loading model from {repo_id}/{subfolder}...")
        model = AutoModel.from_pretrained(
            repo_id, 
            subfolder=subfolder, 
            trust_remote_code=True
        )
        print(f"Successfully loaded {subfolder}")
        print(f"Model type: {type(model)}")
    except Exception as e:
        print(f"Error loading {subfolder}:")
        traceback.print_exc()
