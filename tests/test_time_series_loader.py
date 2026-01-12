from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

import ssm_time_series.data.dataloaders.concat_loader as concat_loader
from ssm_time_series.data.loader import TimeSeriesDataModule


class _DummyDataset(Dataset):
    """Minimal dataset stub that mimics the tuple output of Dataset_Custom."""

    def __init__(self, root_path: str, flag: str, data_path: str, scale: bool = True, **kwargs) -> None:
        self.flag = flag
        self.length = 3

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.length

    def __getitem__(self, index: int):
        seq_x = torch.zeros(384, 1)
        seq_y = torch.zeros(192, 1)
        seq_x_mark = torch.zeros(192, 1)
        seq_y_mark = torch.zeros(192, 1)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


def test_time_series_dataloader_shapes(monkeypatch, tmp_path):
    """Ensure the data module builds loaders and yields tensors with expected shapes."""
    # Arrange: create a fake dataset file so discovery finds something under the path.
    data_dir = tmp_path / "datasets"
    data_dir.mkdir()
    dummy_file = data_dir / "dummy.csv"
    dummy_file.write_text("date,OT\n")

    # Patch Dataset_Custom inside the concatenation helper to avoid reading real data.
    monkeypatch.setattr(concat_loader, "Dataset_Custom", _DummyDataset)

    module = TimeSeriesDataModule(
        dataset_name=dummy_file.name,
        data_dir=str(data_dir),
        batch_size=2,
        val_batch_size=2,
        num_workers=0,
        pin_memory=False,
        normalize=True,
    train_ratio=0.8,
    val_ratio=0.2,
        train=True,
        val=False,
        test=False,
    )

    # Act
    dataset_loaders = module.get_dataloaders()
    assert len(dataset_loaders) == 1
    train_loader = dataset_loaders[0].train
    batch = next(iter(train_loader))

    # Assert
    assert len(batch) == 4
    assert batch[0].shape[1] == 384
    assert batch[2].shape[1] == 192
