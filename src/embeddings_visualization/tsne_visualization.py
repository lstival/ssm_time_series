import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path
import yaml
import sys
import re
import pandas as pd
import training_utils as tu
from time_series_loader import TimeSeriesDataModule
import datasets

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from util import (
    default_device,
    prepare_run_directory,
    load_encoder_checkpoint,
    simple_interpolation,
)
from dataloaders.cronos_dataset import load_chronos_datasets
from moco_training import resolve_path
from down_tasks.forecast_shared import apply_model_overrides

def load_chronos_dataset(dataset_name, repo_id="autogluon/chronos_datasets", split="train", normalize_per_series=True, max_length=512, max_series=1000):
    """
    Load Chronos datasets to extract features.

    Args:
        dataset_name (str): Name of the dataset to load.
        repo_id (str): Repository ID for the dataset.
        split (str): Dataset split to load (e.g., "train", "val").
        normalize_per_series (bool): Whether to normalize each series individually.
        max_length (int): Maximum length of time series to use.
        max_series (int): Maximum number of series to load.

    Returns:
        np.ndarray: Loaded dataset as a numpy array of shape (N, T).
    """
    try:
        print(f"Loading Chronos dataset: {dataset_name}")
        ds = load_chronos_datasets(
            [dataset_name],
            split=split,
            repo_id=repo_id,
            normalize_per_series=normalize_per_series,
            offline_cache_dir="../../data",
            force_offline=True,
        )
        
        sequences = []
        for idx, item in enumerate(ds):
            if idx >= max_series:
                break
                
            target = item.get('target') if isinstance(item, dict) else item
            if isinstance(target, (list, np.ndarray)):
                target = np.array(target, dtype=np.float32)
                # Truncate or pad to max_length
                if len(target) > max_length:
                    target = target[:max_length]
                elif len(target) < max_length:
                    # Pad with zeros
                    padded = np.zeros(max_length, dtype=np.float32)
                    padded[:len(target)] = target
                    target = padded
                
                # Skip sequences with all zeros or constant values
                if np.std(target) > 1e-8:
                    sequences.append(target)
        
        if sequences:
            return np.stack(sequences, axis=0)
        else:
            print(f"Warning: No valid sequences found for {dataset_name}")
            return None
        
    except Exception as e:
        print(f"Error loading Chronos dataset {dataset_name}: {e}")
        return None


def load_icml_dataset(dataset_name, data_dir="../../ICML_datasets", max_length=512, max_series=1000):
    """
    Load ICML datasets from local CSV files.

    Args:
        dataset_name (str): Name of the ICML dataset to load.
        data_dir (str): Path to ICML datasets directory.
        max_length (int): Maximum length of time series to use.
        max_series (int): Maximum number of series to load.

    Returns:
        np.ndarray: Loaded dataset as a numpy array of shape (N, T).
    """
    try:
        print(f"Loading ICML dataset: {dataset_name}")
        dataset_path = Path(data_dir) / dataset_name
        
        if not dataset_path.exists():
            print(f"Warning: ICML dataset path does not exist: {dataset_path}")
            return None
        
        # Find CSV files in the dataset directory
        csv_files = list(dataset_path.glob("*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in {dataset_path}")
            return None
        
        sequences = []
        for csv_file in csv_files:
            try:
                # Load CSV file
                df = pd.read_csv(csv_file)
                
                # Skip date/time columns and use only numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    # Try to find columns that might be numeric but stored as strings
                    for col in df.columns:
                        if col.lower() not in ['date', 'time', 'timestamp']:
                            try:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                if not df[col].isna().all():
                                    numeric_columns = numeric_columns.append(pd.Index([col]))
                            except:
                                continue
                
                if len(numeric_columns) == 0:
                    print(f"Warning: No numeric columns found in {csv_file}")
                    continue
                
                # Extract each numeric column as a separate time series
                for col in numeric_columns:
                    series_data = df[col].fillna(0).values.astype(np.float32)
                    
                    # Truncate or pad to max_length
                    if len(series_data) > max_length:
                        series_data = series_data[:max_length]
                    elif len(series_data) < max_length:
                        padded = np.zeros(max_length, dtype=np.float32)
                        padded[:len(series_data)] = series_data
                        series_data = padded
                    
                    # Skip sequences with all zeros or constant values
                    if np.std(series_data) > 1e-8:
                        sequences.append(series_data)
                        
                        if len(sequences) >= max_series:
                            break
                
                if len(sequences) >= max_series:
                    break
                    
            except Exception as e:
                print(f"Warning: Error reading {csv_file}: {e}")
                continue
        
        if sequences:
            return np.stack(sequences, axis=0)
        else:
            print(f"Warning: No valid sequences found for ICML dataset {dataset_name}")
            return None
        
    except Exception as e:
        print(f"Error loading ICML dataset {dataset_name}: {e}")
        return None


def discover_available_datasets(chronos_data_dir="../../data", icml_data_dir="../../ICML_datasets"):
    """
    Discover all available datasets from both Chronos and ICML sources.
    
    Returns:
        tuple: (chronos_datasets, icml_datasets) - lists of available dataset names
    """
    chronos_datasets = []
    icml_datasets = []
    
    # Discover Chronos datasets from data directory
    try:
        chronos_path = Path(chronos_data_dir)
        if chronos_path.exists():
            chronos_datasets = [d.name for d in chronos_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    except Exception as e:
        print(f"Error discovering Chronos datasets: {e}")
    
    # Discover ICML datasets
    try:
        icml_path = Path(icml_data_dir)
        if icml_path.exists():
            icml_datasets = [d.name for d in icml_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    except Exception as e:
        print(f"Error discovering ICML datasets: {e}")
    
    print(f"Found {len(chronos_datasets)} Chronos datasets: {chronos_datasets[:5]}{'...' if len(chronos_datasets) > 5 else ''}")
    print(f"Found {len(icml_datasets)} ICML datasets: {icml_datasets}")
    
    return chronos_datasets, icml_datasets


def get_encoder_path(config_path):
    """
    Retrieve the encoder path from the configuration file.

    Args:
        config_path (str or Path): Path to the configuration YAML file.

    Returns:
        str: Path to the encoder checkpoint.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If encoder checkpoint path is not found.
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Try to get encoder checkpoint from paths section
    if "paths" in cfg and "encoder_checkpoint" in cfg["paths"]:
        encoder_path = cfg["paths"]["encoder_checkpoint"]
    else:
        raise KeyError("Encoder checkpoint path not found in configuration. Expected 'paths.encoder_checkpoint' key.")

    # Resolve relative paths against config directory
    found_path = Path(encoder_path)
    if not found_path.is_absolute():
        resolved = (cfg_path.parent / found_path).resolve()
    else:
        resolved = found_path

    return str(resolved)


def tsne_visualization(config_path, chronos_datasets=None, icml_datasets=None, repo_id="autogluon/chronos_datasets", 
                      output_dir="results/tsne", max_samples_per_dataset=500, batch_size=128, 
                      chronos_data_dir="../../data", icml_data_dir="../../ICML_datasets"):
    """
    Perform TSNE visualization for embeddings from multiple datasets.

    Args:
        config_path (str): Path to the configuration YAML file.
        chronos_datasets (list): List of Chronos dataset names. If None, auto-discover all.
        icml_datasets (list): List of ICML dataset names. If None, auto-discover all.
        repo_id (str): Repository ID for Chronos datasets.
        output_dir (str): Directory to save the TSNE plots.
        max_samples_per_dataset (int): Max number of samples to use per dataset.
        batch_size (int): Batch size for embedding extraction.
        chronos_data_dir (str): Path to Chronos data directory.
        icml_data_dir (str): Path to ICML datasets directory.
    """
    print(f"Starting t-SNE visualization with datasets: {datasets}")
    
    # Get encoder path from config
    encoder_path = get_encoder_path(config_path)
    print(f"Encoder path: {encoder_path}")

    # Check if encoder checkpoint exists
    checkpoint_path = Path(encoder_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

    # Load configuration files
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        forecast_cfg = yaml.safe_load(fh) or {}

    model_section = dict(forecast_cfg.get("model") or {})
    model_config_candidate = model_section.get("config")
    if model_config_candidate is None:
        raise ValueError("Configuration missing required key 'model.config'.")

    model_config_path = Path(model_config_candidate)
    if not model_config_path.is_absolute():
        model_config_path = (cfg_path.parent / model_config_path).resolve()

    base_config = tu.load_config(model_config_path)
    overrides = dict(model_section.get("overrides") or {})

    model_cfg = apply_model_overrides(
        base_config.model,
        token_size=overrides.get("token_size"),
        model_dim=overrides.get("model_dim"),
        embedding_dim=overrides.get("embedding_dim"),
        depth=overrides.get("depth"),
    )

    # Build and load encoder
    device = default_device()
    print(f"Using device: {device}")
    
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    print(f"Loading encoder checkpoint: {checkpoint_path}")
    load_encoder_checkpoint(encoder, checkpoint_path, device)
    encoder.eval()

    # Auto-discover datasets if not provided
    if chronos_datasets is None or icml_datasets is None:
        discovered_chronos, discovered_icml = discover_available_datasets(chronos_data_dir, icml_data_dir)
        if chronos_datasets is None:
            chronos_datasets = discovered_chronos
        if icml_datasets is None:
            icml_datasets = discovered_icml
    
    all_datasets = [(name, "chronos") for name in chronos_datasets] + [(name, "icml") for name in icml_datasets]
    print(f"Processing {len(all_datasets)} total datasets: {len(chronos_datasets)} Chronos + {len(icml_datasets)} ICML")

    all_embeddings = []
    all_labels = []
    dataset_names = []

    # Process each dataset
    for label, (dataset_name, dataset_type) in enumerate(all_datasets):
        print(f"Processing {dataset_type} dataset: {dataset_name}")

        # Load dataset based on type
        if dataset_type == "chronos":
            data = load_chronos_dataset(dataset_name, repo_id, max_length=512, max_series=max_samples_per_dataset)
        else:  # icml
            data = load_icml_dataset(dataset_name, icml_data_dir, max_length=512, max_series=max_samples_per_dataset)
        
        if data is None:
            print(f"Warning: dataset {dataset_name} returned None, skipping.")
            continue

        print(f"Dataset {dataset_name} shape: {data.shape}")
        
        # Limit samples
        n_samples = min(len(data), max_samples_per_dataset)
        data = data[:n_samples]
        print(f"Using {n_samples} samples from {dataset_name}")

        embeddings_list = []
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = data[start:end]
                
                # Convert to tensor
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
                
                # Add channel dimension if needed (B, T) -> (B, 1, T)
                if batch_tensor.dim() == 2:
                    batch_tensor = batch_tensor.unsqueeze(1)
                
                # Get embeddings
                emb = encoder(batch_tensor)
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                
                emb = emb.detach().cpu().numpy()
                embeddings_list.append(emb)

        if len(embeddings_list) == 0:
            print(f"No embeddings extracted for {dataset_name}")
            continue

        embeddings = np.concatenate(embeddings_list, axis=0)
        print(f"Embeddings shape for {dataset_name}: {embeddings.shape}")
        
        all_embeddings.append(embeddings)
        all_labels.extend([label] * len(embeddings))
        dataset_names.append(f"{dataset_name} ({dataset_type})")

    if len(all_embeddings) == 0:
        raise RuntimeError("No embeddings produced for any dataset.")

    # Combine all embeddings and labels
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)
    print(f"Total embeddings shape: {all_embeddings.shape}")

    # Perform TSNE
    print("Running t-SNE...")
    perplexity = min(30, len(all_embeddings) // 3)  # Adjust perplexity based on sample size
    tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=perplexity, max_iter=1000)
    tsne_results = tsne.fit_transform(all_embeddings)
    print("t-SNE completed")

    # Plot TSNE
    plt.figure(figsize=(15, 12))
    colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_names)))
    
    for label, dataset_name in enumerate(dataset_names):
        indices = np.where(all_labels == label)[0]
        if indices.size == 0:
            continue
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                   label=f"{dataset_name} (n={len(indices)})", 
                   alpha=0.7, s=20, c=[colors[label]])

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE Visualization of Time Series Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    encoder_name = Path(encoder_path).stem
    plot_path = os.path.join(output_dir, f"tsne_{encoder_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"t-SNE plot saved to {plot_path}")


# Example usage
if __name__ == "__main__":
    config_path = r"c:\WUR\ssm_time_series\src\configs\chronos_forecast.yaml"
    
    try:
        # Use all available datasets from both sources
        tsne_visualization(
            config_path=config_path,
            chronos_datasets=None,  # Auto-discover all Chronos datasets
            icml_datasets=None,     # Auto-discover all ICML datasets
            repo_id="autogluon/chronos_datasets",
            output_dir="results/tsne",
            max_samples_per_dataset=200,  # Reduced for faster processing with more datasets
            batch_size=64,
            chronos_data_dir="../../data",
            icml_data_dir="../../ICML_datasets"
        )
    except Exception as e:
        print(f"Error during t-SNE visualization: {e}")
        import traceback
        traceback.print_exc()