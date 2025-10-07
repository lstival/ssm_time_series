import os
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import Subset

class TimeSeriesDataModule:
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        batch_size: int = 128,
        val_batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        # new flags to request specific splits from Dataset_Custom
        train: bool = True,
        val: bool = True,
        test: bool = False,
    ):
        assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must equal 1.0"
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.train = train
        self.val = val
        self.test = test
        self._built = False

    def _make_dataset(self, flag: str):
        # Try passing normalize to Dataset_Custom if it accepts it, otherwise fall back.
        try:
            return Dataset_Custom(root_path=self.data_dir, flag=flag, data_path=self.dataset_name, normalize=self.normalize)
        except TypeError:
            return Dataset_Custom(root_path=self.data_dir, flag=flag, data_path=self.dataset_name)

    def setup(self):
        # If the underlying Dataset_Custom supports explicit flags, use them.
        # Otherwise, fall back to creating a single "train" dataset and splitting.
        # First attempt to instantiate per-flag datasets to check support.
        supports_flags = True
        try:
            _ = self._make_dataset("train")
        except Exception:
            supports_flags = False

        if supports_flags:
            # create requested splits directly via Dataset_Custom flag
            self.train_dataset = self._make_dataset("train") if self.train else None
            self.val_dataset = self._make_dataset("val") if self.val else None
            self.test_dataset = self._make_dataset("test") if self.test else None
        else:
            # fall back: load full dataset (using whatever flag 'train' expects) and split by indices
            full = self._make_dataset("train")
            n = len(full)
            split = int(self.train_ratio * n)
            # train/val split from the full dataset
            self.train_dataset = Subset(full, range(0, split)) if self.train else None
            self.val_dataset = Subset(full, range(split, n)) if self.val else None
            self.test_dataset = None  # no explicit test split available in fallback

        self._built = True

    def get_dataloaders(self):
        if not self._built:
            self.setup()

        train_loader = (
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            if self.train_dataset is not None
            else None
        )

        val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            if self.val_dataset is not None
            else None
        )

        test_loader = (
            DataLoader(
                self.test_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            if self.test_dataset is not None
            else None
        )

        # Keep backward compatibility: if test was requested return 3-tuple, else 2-tuple
        if self.test:
            return train_loader, val_loader, test_loader
        return train_loader, val_loader

if __name__ == '__main__':
    # Example usage with the requested settings
    data_module = TimeSeriesDataModule(
        dataset_name="ETTh1.csv",
        data_dir="../ICML_datasets/ETT-small",
        # data_dir="ICML_datasets/ETT-small",
        batch_size=128,
        val_batch_size=256,
        num_workers=4,
        pin_memory=True,
        normalize=True,
        train_ratio=0.8,
        val_ratio=0.2,
        train=True,
        val=True,
        test=False,
    )

    train_loader, val_loader = data_module.get_dataloaders()
    print("Train batches:", len(train_loader), "Val batches:", len(val_loader))