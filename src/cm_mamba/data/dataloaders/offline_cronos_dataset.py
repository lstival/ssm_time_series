import os
from typing import Iterable, Optional

from ssm_time_series.data.dataloaders.cronos_dataset import load_chronos_datasets


def load_chronos_datasets_offline(
    dataset_names: Iterable[str],
    *,
    split: str = "train",
    offline_cache_dir: Optional[str] = None,
    **kwargs,
):
    """Load Chronos datasets from a custom offline cache directory.

    Args:
        dataset_names: Dataset names to load.
        split: Dataset split (default: "train").
        offline_cache_dir: Offline cache directory path. If ``None``, uses
            ``HF_DATASETS_CACHE`` or ``SSM_HF_OFFLINE_CACHE_DIR``.
        **kwargs: Additional arguments passed to ``load_chronos_datasets``.

    Returns:
        Combined dataset.
    """
    cache_dir = (
        offline_cache_dir
        or os.environ.get("SSM_HF_OFFLINE_CACHE_DIR")
        or os.environ.get("HF_DATASETS_CACHE")
    )
    if cache_dir is None:
        raise ValueError("offline_cache_dir must be provided or set via SSM_HF_OFFLINE_CACHE_DIR")

    original_cache = os.environ.get("HF_DATASETS_CACHE")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        return load_chronos_datasets(dataset_names=dataset_names, split=split, **kwargs)
    finally:
        if original_cache is not None:
            os.environ["HF_DATASETS_CACHE"] = original_cache
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)