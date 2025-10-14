import datasets
import pandas as pd

ds = datasets.load_dataset("autogluon/chronos_datasets", "m4_daily", split="train")
# ds = datasets.load_dataset("autogluon/chronos_datasets", "weatherbench_daily", keep_in_memory=False, split="train")
# ds = datasets.load_dataset("autogluon/chronos_datasets", "weatherbench_hourly_temperature", streaming=True, split="train")
ds.set_format("numpy")  # sequences returned as numpy arrays


def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
    """Convert dataset to long data frame format."""
    sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
    return ds.to_pandas().explode(sequence_columns).infer_objects()

print(to_pandas(ds).head())


