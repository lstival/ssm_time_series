import torch
import numpy as np
from cm_mamba.data.loader import SimpleTimeSeriesDataModule

def custom_dataset_demo():
    # 1. Generate fake data (e.g., from a CSV or Numpy array)
    # 10,000 timesteps, 15 features
    data_array = np.sin(np.linspace(0, 100, 10000)).reshape(-1, 1) * np.ones((1, 15))
    data_tensor = torch.from_numpy(data_array).float()
    
    # 2. Setup DataModule
    datamodule = SimpleTimeSeriesDataModule(
        data=data_tensor,
        batch_size=32,
        seq_len=96,
        pred_len=720,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    # 3. Get Loaders
    train_loader, val_loader, test_loader = datamodule.get_loaders()
    
    # 4. Iterate
    x, y = next(iter(train_loader))
    print(f"Input batch shape: {x.shape}")    # (batch, seq_len, features)
    print(f"Target batch shape: {y.shape}")   # (batch, pred_len, features)

if __name__ == "__main__":
    custom_dataset_demo()
