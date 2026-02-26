import torch
from forecasting import CM_MambaForecastModel, CM_MambaForecastConfig

# Load config and model using the local file
config = CM_MambaForecastConfig.from_pretrained("./path_to_local_folder")
model = CM_MambaForecastModel(config)
model.eval()

# Check shape [Batch, Time, Features]
x = torch.randn(1, 128, 1) # Use shape matching your model
with torch.no_grad():
    out = model(x)
print(f"Success! Output shape: {out.shape}")