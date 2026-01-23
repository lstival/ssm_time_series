
import torch
from transformers import AutoModel, AutoConfig
import os

repo_id = "lstival/CM-Mamba"
subfolders = ["cm-mamba-tiny", "cm-mamba-mini"]

for subfolder in subfolders:
    print(f"\n--- Testing {subfolder} ---")
    try:
        # We try to load it without trust_remote_code first, and then with it.
        # But since it's a custom model, it MUST have trust_remote_code=True
        model = AutoModel.from_pretrained(
            repo_id, 
            subfolder=subfolder, 
            trust_remote_code=True
        )
        print(f"Successfully loaded {subfolder}")
    except Exception as e:
        print(f"Error loading {subfolder}: {e}")
