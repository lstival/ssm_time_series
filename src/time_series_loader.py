import os
from typing import List, Optional

from torch.utils.data import DataLoader
from dataloaders import DatasetLoaders, build_dataset_loader_list
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
        # train_ratio: float = 0.8,
        # val_ratio: float = 0.2,
        filename: Optional[str] = None,
        # new flags to request specific splits from Dataset_Custom
        train: bool = True,
        val: bool = True,
        test: bool = False,
    ):
        # assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must equal 1.0"
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        # self.train_ratio = train_ratio
        # self.val_ratio = val_ratio
        self.train = train
        self.val = val
        self.test = test
        self._built = False
        self.filename = filename
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.train_loaders: List[DataLoader] = []
        self.val_loaders: List[DataLoader] = []
        self.test_loaders: List[DataLoader] = []
        self.dataset_loaders: List[DatasetLoaders] = []

    def setup(self):
        dataset_files = discover_dataset_files(self.data_dir, filename=self.filename)
        if self.dataset_name:
            dataset_files = {
                key: path
                for key, path in dataset_files.items()
                if os.path.basename(path) == self.dataset_name or key == self.dataset_name
            }
            if not dataset_files:
                raise FileNotFoundError(
                    f"Dataset '{self.dataset_name}' not found under '{self.data_dir}'."
                )

        dataset_loaders = build_dataset_loader_list(
            self.data_dir,
            batch_size=self.batch_size,
            val_batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            normalize=self.normalize,
            # train_ratio=self.train_ratio,
            # val_ratio=self.val_ratio,
            include_train=self.train,
            include_val=self.val,
            include_test=self.test,
            dataset_files=dataset_files,
            filename=self.filename,
        )

        self.dataset_loaders = dataset_loaders
        self.train_loaders = [entry.train for entry in dataset_loaders if entry.train is not None]
        self.val_loaders = [entry.val for entry in dataset_loaders if entry.val is not None]
        self.test_loaders = [entry.test for entry in dataset_loaders if entry.test is not None]
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self._built = True

    def get_dataloaders(self):
        if not self._built:
            self.setup()

        return self.dataset_loaders

    def get_train_dataloaders(self) -> List[DataLoader]:
        if not self._built:
            self.setup()
        return self.train_loaders

    def get_val_dataloaders(self) -> List[DataLoader]:
        if not self._built:
            self.setup()
        return self.val_loaders

    def get_test_dataloaders(self) -> List[DataLoader]:
        if not self._built:
            self.setup()
        return self.test_loaders

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Example usage with the requested settings
    data_root = "../ICML_datasets"
    dataset_name = "PEMS07.npz"
    # dataset_name = "solar_AL.txt"
    # dataset_name = "ETTh1.csv"

    module = TimeSeriesDataModule(data_dir=data_root, dataset_name=dataset_name, batch_size=1, train=False, val=False, test=True)
    loaders = module.get_dataloaders()
    for dataset in loaders:
        if dataset.train is not None:
            print(f"{dataset.name}: {len(dataset.train)} train batches")
        if dataset.val is not None:
            print(f"{dataset.name}: {len(dataset.val)} val batches")
        if dataset.test is not None:
            print(f"{dataset.name}: {len(dataset.test)} test batches")
    
    aa = next(iter(dataset.test))
    print(aa[0].shape)

    plt.plot(np.arange(aa[0].shape[1]), aa[0][0,:,0])
    plt.show()
    plt.plot(np.arange(aa[1][0,:720,0].shape[0]), aa[1][0,:720,0])
    plt.show()