import os
from typing import Optional

from torch.utils.data import DataLoader
from dataloaders import build_concat_dataloaders
from dataloaders.utils import discover_dataset_files

class TimeSeriesDataModule:
    def __init__(
        self,
        dataset_name: str = "",
        data_dir: str = "",
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
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def setup(self):
        dataset_files = discover_dataset_files(self.data_dir)
        if self.dataset_name:
            dataset_files = {
                key: path
                for key, path in dataset_files.items()
                if os.path.basename(path) == self.dataset_name
            }
            if not dataset_files:
                raise FileNotFoundError(
                    f"Dataset '{self.dataset_name}' not found under '{self.data_dir}'."
                )

        train_loader, val_loader, test_loader = build_concat_dataloaders(
            self.data_dir,
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            normalize=self.normalize,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            include_train=self.train,
            include_val=self.val,
            include_test=self.test,
            # dataset_files=dataset_files,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self._built = True

    def get_dataloaders(self):
        if not self._built:
            self.setup()

        # Keep backward compatibility: if test was requested return 3-tuple, else 2-tuple
        if self.test:
            return self.train_loader, self.val_loader, self.test_loader
        return self.train_loader, self.val_loader

if __name__ == '__main__':
    # Example usage with the requested settings
    data_root = "../ICML_datasets"
    dataset_files = discover_dataset_files(data_root)

    if not dataset_files:
        print(f"No dataset files found under {data_root}.")
    else:
        print("Datasets discovered:", list(dataset_files.keys()))

        batch_size = 128
        val_batch_size = 256
        num_workers = 4
        pin_memory = True
        normalize = True
        train_ratio = 0.8
        val_ratio = 0.2
        include_train = True
        include_val = True
        include_test = False

        train_loader, val_loader, test_loader = build_concat_dataloaders(
            data_root,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            normalize=normalize,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            include_train=include_train,
            include_val=include_val,
            include_test=include_test,
        )

        if train_loader is not None:
            print("Combined train batches:", len(train_loader))
        if val_loader is not None:
            print("Combined val batches:", len(val_loader))
        if test_loader is not None:
            print("Combined test batches:", len(test_loader))