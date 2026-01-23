
import torch
from transformers import AutoModel
import sys
import os

# We do NOT append src to path here, assuming the user installed the package or the remote code handles it.
# Ideally, "trust_remote_code=True" should handle it IF the repo contains everything.
# But since the repo code imports 'ssm_time_series', we need it installed.
# For verification in THIS environment, we add src if not installed.
try:
    import ssm_time_series
except ImportError:
    print("Package not installed, adding src to path for testing purposes.")
    sys.path.append(os.path.abspath("src"))

repo_id = "lstival/CM-Mamba"
subfolders = ["cm-mamba-tiny", "cm-mamba-mini"]

for subfolder in subfolders:
    print(f"\n{'='*20} Testing {subfolder} {'='*20}", flush=True)
    try:
        model = AutoModel.from_pretrained(
            repo_id, 
            subfolder=subfolder, 
            trust_remote_code=True,
            force_download=True
        )
        print(f"Successfully loaded {subfolder}")
        
        # Print encoder input projection to verify shapes
        if hasattr(model, 'encoder'):
            print(f"Encoder Input Proj: {model.encoder.input_proj}")
            print(f"Encoder Token Size: {model.encoder.token_size}")
            print(f"Encoder Input Dim: {model.encoder.input_dim}")
        
        # Dummy Input: [Batch, Time, Features]
        # Config says input_dim=32
        B, T, F = 1, 96, 32 
        x = torch.randn(B, T, F)
        
        print(f"Running forward pass with input {x.shape}...")
        y = model(x)
        print(f"Output shape: {y.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        # import traceback
        # traceback.print_exc()
