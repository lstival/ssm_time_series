import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Union

class TimeSeriesDataset(Dataset):
    """
    A simple, generic time-series dataset.
    Expects data as (N, Features) where N is the total number of timesteps.
    """
    def __init__(
        self, 
        data: torch.Tensor, 
        seq_len: int = 96, 
        pred_len: int = 720,
        stride: int = 1
    ):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        
        # Total window size = seq_len + pred_len
        self.window_size = seq_len + pred_len
        self.length = (len(data) - self.window_size) // stride + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        return seq_x, seq_y

class SimpleTimeSeriesDataModule:
    """
    A minimal DataModule to handle train/val/test splits.
    """
    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int = 32,
        seq_len: int = 96,
        pred_len: int = 720,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        num_workers: int = 0
    ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_workers = num_workers
        
        n = len(data)
        self.train_end = int(n * train_ratio)
        self.val_end = int(n * (train_ratio + val_ratio))
        
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data = self.data[:self.train_end]
        val_data = self.data[self.train_end:self.val_end]
        test_data = self.data[self.val_end:]
        
        train_set = TimeSeriesDataset(train_data, self.seq_len, self.pred_len)
        val_set = TimeSeriesDataset(val_data, self.seq_len, self.pred_len)
        test_set = TimeSeriesDataset(test_data, self.seq_len, self.pred_len)
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return train_loader, val_loader, test_loader
