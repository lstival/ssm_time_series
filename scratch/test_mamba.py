import torch
from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing Mamba on {device}")

model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2).to(device)
x = torch.randn(2, 64, 128).to(device)

try:
    y = model(x)
    print(f"Success! Output shape: {y.shape}")
except Exception as e:
    print(f"Failed! Error: {e}")
