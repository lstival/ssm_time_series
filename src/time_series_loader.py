import os
from typing import List, Optional, Sequence, Tuple, Union

from torch.utils.data import DataLoader
from dataloaders import DatasetLoaders, build_dataset_loader_list
from dataloaders.utils import discover_dataset_files
from data_provider.data_loader import LABEL_LEN, PRED_LEN

class TimeSeriesDataModule:
    def __init__(
        self,
        dataset_name: str = "",
        dataset_names: Optional[List[str]] = None,
        data_dir: str = "",
        batch_size: int = 128,
        val_batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        normalize: bool = True,
        # train_ratio: float = 0.8,
        # val_ratio: float = 0.2,
        filename: Optional[str] = None,
        sample_size: Optional[Union[int, Sequence[int]]] = None,
        # new flags to request specific splits from Dataset_Custom
        train: bool = True,
        val: bool = True,
        test: bool = False,
    ):
        # assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must equal 1.0"
        self.dataset_name = dataset_name
        self.dataset_names = dataset_names
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize
        # self.train_ratio = train_ratio
        # self.val_ratio = val_ratio
        self.sample_size = self._normalize_sample_size(sample_size)
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
        
        # Filter by specific dataset names if provided
        if self.dataset_names:
            dataset_files = {
                key: path
                for key, path in dataset_files.items()
                if any(os.path.basename(path) == name or key == name for name in self.dataset_names)
            }
            if not dataset_files:
                raise FileNotFoundError(
                    f"None of the specified datasets {self.dataset_names} found under '{self.data_dir}'."
                )
        elif self.dataset_name:
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
            sample_size=self.sample_size,
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

    @staticmethod
    def _normalize_sample_size(
        sample_size: Optional[Union[int, Sequence[int]]]
    ) -> Optional[Tuple[int, int, int]]:
        if sample_size is None:
            return None

        if isinstance(sample_size, (str, bytes)):
            raise TypeError("sample_size must be an int or a sequence of ints, not a string")

        if isinstance(sample_size, int):
            seq_len = sample_size
            label_len = LABEL_LEN
            pred_len = PRED_LEN
        else:
            values = list(sample_size)
            if not values:
                raise ValueError("sample_size sequence cannot be empty")
            seq_len = values[0]
            label_len = values[1] if len(values) > 1 else LABEL_LEN
            pred_len = values[2] if len(values) > 2 else PRED_LEN

        seq_len = int(seq_len)
        label_len = int(label_len)
        pred_len = int(pred_len)

        if seq_len <= 0:
            raise ValueError("sequence length in sample_size must be positive")
        if label_len < 0:
            raise ValueError("label length in sample_size cannot be negative")
        if pred_len <= 0:
            raise ValueError("prediction length in sample_size must be positive")

        return seq_len, label_len, pred_len

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Example usage with the requested settings
    data_root = "../ICML_datasets"
    # dataset_name = "PEMS03.npz"
    # dataset_name = "solar_AL.txt"
    dataset_name = "ETTm1.csv"
    # dataset_name = "electricity.csv"

    module = TimeSeriesDataModule(
        data_dir=data_root,
        dataset_name=dataset_name,
        batch_size=1,
        train=True,
        val=True,
        test=True,
        sample_size=48,  # override default 96-step context to 48 steps
    )
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

    inp = aa[0][0, :, 0]
    tgt = aa[1][0, :, 0]
    inp_x = np.arange(inp.shape[0])
    tgt_x = np.arange(inp.shape[0], inp.shape[0] + tgt.shape[0])

    plt.figure()
    plt.plot(inp_x, inp, color='blue', label='input 0')
    plt.plot(tgt_x, tgt, color='orange', label='target 1)')
    plt.xlabel('time step')
    plt.legend()
    plt.show()