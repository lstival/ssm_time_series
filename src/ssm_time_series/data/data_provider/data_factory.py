from ssm_time_series.data.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
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

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


if __name__ == "__main__":
    # data_root = "../ICML_datasets/ETT-small"
    data_root = r"C:\WUR\ssm_time_series\ICML_datasets\ETT-small"
    # data_root = r"C:\WUR\ssm_time_series\ICML_datasets\PEMS"
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
