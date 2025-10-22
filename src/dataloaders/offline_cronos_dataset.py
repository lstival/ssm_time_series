import datasets
import os
from .cronos_dataset import load_chronos_datasets

def load_chronos_datasets_offline(
    dataset_names,
    split="train",
    offline_cache_dir="D:/my_datasets",  # Custom cache directory
    **kwargs
):
    """
    Load Chronos datasets from a custom offline cache directory.
    
    Args:
        dataset_names: List of dataset names to load
        split: Dataset split (default: "train")
        offline_cache_dir: Path to the offline cache directory
        **kwargs: Additional arguments passed to load_chronos_datasets
    
    Returns:
        Combined dataset
    """
    # Set the cache directory for this session
    original_cache = os.environ.get('HF_DATASETS_CACHE')
    os.environ['HF_DATASETS_CACHE'] = offline_cache_dir
    
    try:
        # Load datasets using the original function
        dataset = load_chronos_datasets(
            dataset_names=dataset_names,
            split=split,
            **kwargs
        )
        return dataset
    finally:
        # Restore original cache setting
        if original_cache:
            os.environ['HF_DATASETS_CACHE'] = original_cache
        else:
            os.environ.pop('HF_DATASETS_CACHE', None)

# Example usage:
if __name__ == "__main__":
    # List of datasets you have cached
    datasets_to_load = [
        "m4_daily",
        "m4_hourly", 
        "m4_monthly",
        "m4_yearly",
        "exchange_rate",
        # Add other datasets you have cached
    ]
    
    # Load from offline cache
    dataset = load_chronos_datasets_offline(
        dataset_names=datasets_to_load,
        offline_cache_dir="D:/my_datasets"  # Adjust path as needed
    )
    
    print(f"Loaded {len(dataset)} samples from offline cache")