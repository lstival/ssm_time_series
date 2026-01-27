import torch
from transformers import AutoModel

# Load the model
model_id = "lstival/cm-mamba-tiny"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
model.eval()

# Input shape: [Batch, Time, Features]
# CM-Mamba expects 32 features by default
dummy_input = torch.randn(1, 384, 32) 

with torch.no_grad():
    forecast = model(dummy_input)

print(f"Forecast shape: {forecast.shape}") # [1, 720, 1]
