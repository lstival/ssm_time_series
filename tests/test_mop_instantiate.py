import torch
import torch.nn as nn
import sys
from pathlib import Path

print("Unit Test Started")
from models.mop_forecast import MoPForecastModel

def test():
    # Mock encoders
    class MockEncoder(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.embedding_dim = dim
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, self.embedding_dim)

    enc = MockEncoder(64)
    vis = MockEncoder(64)
    
    # Try different configs
    configs = [
        {"norm_mode": "identity", "head_type": "linear"},
        {"norm_mode": "revin", "head_type": "mlp", "use_ln_head": True},
        {"norm_mode": "minmax", "head_type": "linear", "learnable_scale": True},
        {"norm_mode": "identity", "head_type": "linear", "scale_cond": True, "temperature": 0.5},
    ]
    
    for cfg in configs:
        print(f"Testing config: {cfg}")
        model = MoPForecastModel(
            encoder=enc, visual_encoder=vis, input_dim=128,
            horizons=[96, 192], **cfg
        )
        x = torch.randn(4, 1, 336) # (B, C, L)
        pred = model(x, 96)
        print(f"  Success! Pred shape: {pred.shape}")
        
        # Test greedy predict
        pred_greedy = model.greedy_predict(torch.randn(4, 1, 336), 192, 336)
        print(f"  Greedy Success! Pred shape: {pred_greedy.shape}")

if __name__ == "__main__":
    test()
