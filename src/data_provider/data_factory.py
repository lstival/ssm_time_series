from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from torch.utils.data import DataLoader
from types import SimpleNamespace
import torch
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    base_freq = getattr(args, 'freq', 'h')
    num_workers = getattr(args, 'num_workers', 0)
    pin_memory = getattr(args, 'pin_memory', True)
    persistent_workers = getattr(args, 'persistent_workers', False)
    if num_workers <= 0:
        persistent_workers = False

    if flag == 'test':
        shuffle_flag = getattr(args, 'test_shuffle', False)
        drop_last = getattr(args, 'test_drop_last', True)
        batch_size = getattr(args, 'test_batch_size', 1)
        freq = base_freq
    elif flag == 'pred':
        shuffle_flag = getattr(args, 'pred_shuffle', False)
        drop_last = getattr(args, 'pred_drop_last', False)
        batch_size = getattr(args, 'pred_batch_size', 1)
        freq = base_freq
        Data = Dataset_Pred
    else:
        freq = base_freq
        if flag == 'val':
            batch_size = getattr(args, 'val_batch_size', getattr(args, 'batch_size', 1))
            shuffle_flag = getattr(args, 'val_shuffle', True)
            drop_last = getattr(args, 'val_drop_last', True)
        else:
            batch_size = getattr(args, 'batch_size', 1)
            shuffle_flag = getattr(args, 'train_shuffle', True)
            drop_last = getattr(args, 'train_drop_last', True)

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        scale=getattr(args, 'scale', True),
        scaler_type=getattr(args, 'scaler_type', 'minmax'),
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return data_set, data_loader


if __name__ == "__main__":
    # data_root = "../ICML_datasets/ETT-small"
    data_root = r"c:/WUR/CM-Mamba\ICML_datasets\ETT-small"
    # data_root = r"c:/WUR/CM-Mamba\ICML_datasets\PEMS"
    # dataset_name = "PEMS07.npz"
    # dataset_name = "solar_AL.txt"
    dataset_name = "ETTh1.csv"

    args = SimpleNamespace(
        root_path=data_root,
        data_path=dataset_name,
        data='ETTh1',      # adjust to match dataset_name (e.g. 'ETTh1', 'ETTm1', 'PEMS', 'Solar', etc.)
        embed='timeF',     # or other embedding option your code supports
        batch_size=32,
        num_workers=0,
        seq_len=384,
        label_len=64,
        pred_len=720,
        features='S',      # 'M' for multivariate, 'S' for single, adjust as needed
        target='OT',       # column name to predict (adjust to your dataset)
        freq='h'           # data frequency ('h' for hourly, 't' for minute, etc.)
    )

    for split in ('train', 'val', 'test'):
        dataset, loader = data_provider(args, split)
        print(f"{split}: dataset size={len(dataset)}, loader batches={len(loader)}")

        try:
            sample = next(iter(loader))
        except Exception as e:
            print(f"  Could not fetch a sample from loader: {e}")
            continue

        # Print shapes/types for the fetched sample
        if isinstance(sample, (list, tuple)):
            for i, s in enumerate(sample):
                if hasattr(s, "shape"):
                    print(f"  sample[{i}] shape = {s.shape}")
                elif torch.is_tensor(s):
                    print(f"  sample[{i}] tensor shape = {s.size()}")
                elif isinstance(s, np.ndarray):
                    print(f"  sample[{i}] ndarray shape = {s.shape}")
                else:
                    print(f"  sample[{i}] type = {type(s)}, repr = {repr(s)[:100]}")
        else:
            if hasattr(sample, "shape"):
                print(f"  sample shape = {sample.shape}")
            elif torch.is_tensor(sample):
                print(f"  sample tensor shape = {sample.size()}")
            elif isinstance(sample, np.ndarray):
                print(f"  sample ndarray shape = {sample.shape}")
            else:
                print(f"  sample type = {type(sample)}, repr = {repr(sample)[:100]}")
