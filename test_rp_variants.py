import torch
import sys
from pathlib import Path

# Add src to path
src_path = Path("src").resolve()
sys.path.append(str(src_path))
sys.path.append(str(src_path / "models"))

from models.mamba_visual_encoder import MambaVisualEncoder

def test_rp_variants():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")
    
    dummy_ts = torch.randn(2, 307, 96).to(device)
    
    variants = ["correct", "shuffled", "random"]
    
    for variant in variants:
        print(f"\n--- Testing RP Mode: {variant} ---")
        try:
            encoder = MambaVisualEncoder(
                input_dim=32,
                model_dim=128,
                embedding_dim=64,
                depth=2,
                rp_mode=variant
            ).to(device)
            
            output = encoder(dummy_ts)
            print(f"Output shape: {output.shape}")
            assert output.shape == (2, 64)
            print("Forward pass successful!")
        except Exception as e:
            print(f"Error in variant {variant}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_rp_variants()
