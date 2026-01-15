import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Union
from sklearn.preprocessing import StandardScaler

class TFBDataset(Dataset):
    """
    Dataset loader for TFB-formatted CSV files.
    The format is expected to be 'date, data, cols' (melted format).
    """
    def __init__(
        self,
        file_path: str,
        context_length: int = 96,
        prediction_length: int = 96,
        stride: int = 1,
        target_col: Optional[str] = None,
        normalize: bool = True,
    ):
        self.file_path = file_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.normalize = normalize
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Ensure correct columns exist
        required_cols = {"date", "data", "cols"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV {file_path} must contain columns {required_cols}")
            
        # Group by 'cols' to separate different series (univariate or multivariate components)
        self.series_data = []
        unique_cols = df["cols"].unique()
        
        # If target_col is specified, we only use that one. 
        # Otherwise, we treat each 'col' as a separate series (which follows the averaging logic in the dual encoder script)
        if target_col is not None:
            if target_col not in unique_cols:
                raise ValueError(f"target_col '{target_col}' not found in {file_path}")
            unique_cols = [target_col]
            
        for col_name in unique_cols:
            col_data = df[df["cols"] == col_name]["data"].values.astype(np.float32)
            self.series_data.append(col_data)
            
        # Stack into (num_series, time_steps)
        # We assume all series in the same file have the same length
        self.data = np.stack(self.series_data, axis=0) # (C, L)
        
        # Interpolation for short series
        min_length = context_length + prediction_length
        current_length = self.data.shape[1]
        if current_length < min_length:
            print(f"  [TFBDataset] Series too short ({current_length} < {min_length}). Interpolating...")
            # Use linear interpolation to upscale to min_length
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, min_length)
            
            # Interpolate each channel
            new_data = np.zeros((self.data.shape[0], min_length), dtype=self.data.dtype)
            for c in range(self.data.shape[0]):
                new_data[c, :] = np.interp(x_new, x_old, self.data[c, :])
            self.data = new_data

        # Scaling
        self.scaler = StandardScaler()
        self.means = None # Initialize means and stds for consistency
        self.stds = None
        if self.normalize:
            # Fit scaler on (time_steps, channels) for StandardScaler
            # Then transform back to (channels, time_steps)
            data_reshaped = self.data.T # (L, C)
            self.scaler.fit(data_reshaped)
            self.data = self.scaler.transform(data_reshaped).T # (C, L)
            self.means = self.scaler.mean_.reshape(-1, 1) # Store means as (C, 1)
            self.stds = np.sqrt(self.scaler.var_).reshape(-1, 1) # Store stds as (C, 1) + 1e-8 for safety

        # Calculate window indices
        self.total_len = self.data.shape[1]
        self.window_indices = []
        max_start = self.total_len - self.context_length - self.prediction_length
        for start in range(0, max_start + 1, self.stride):
            self.window_indices.append(start)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.window_indices[idx]
        mid = start + self.context_length
        end = mid + self.prediction_length
        
        # data is (C, L)
        seq_x = self.data[:, start:mid] # (C, context_len)
        seq_y = self.data[:, mid:end]   # (C, pred_len)
        
        # Return as (channels, sequence, features) where features=1 for now as per TFB format
        # Actually, the dual encoder expects (batch, channels, sequence, features)
        # or (batch, sequence, features)
        
        # For simplicity and compatibility with ICML_zeroshot_forecast_dual.py, 
        # we return (sequence, channels) and let the evaluator handle it.
        # Wait, the reference script says: 
        # seq_x shape: (batch, sequence, features)
        # If multi-channel: (batch, channels, sequence, features)
        
        # Let's return (sequence, channels) which is standard for most loaders in this repo
        return torch.from_numpy(seq_x.T), torch.from_numpy(seq_y.T)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        data: (..., channels)
        """
        if not self.normalize:
            return data
        
        # means/stds are (C, 1). Transpose to (1, C) for broadcasting with (..., C)
        return data * self.stds.T + self.means.T

def get_tfb_dataloader(
    file_path: str,
    batch_size: int = 32,
    context_length: int = 96,
    prediction_length: int = 96,
    stride: int = 1,
    shuffle: bool = False,
    num_workers: int = 4,
) -> Tuple[DataLoader, TFBDataset]:
    dataset = TFBDataset(
        file_path=file_path,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, dataset
