"""Utility to calculate Silhouette scores for time series embeddings."""

import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Dict, Optional, Union, List
import torch

def compute_silhouette(embeddings: Union[np.ndarray, torch.Tensor], labels: Optional[Union[np.ndarray, torch.Tensor]] = None) -> float:
    """
    Compute the mean Silhouette Coefficient for the given embeddings and labels.
    
    If labels are not provided, this function will return 0.0 or raise an error 
    depending on the context. In this experiment, labels are mandatory.
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
        
    if labels is None:
        raise ValueError("Labels are required to compute Silhouette score.")
        
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
        
    return float(silhouette_score(embeddings, labels))

def get_periodic_labels(timestamps: List[int], period: int = 24) -> np.ndarray:
    """Helper to generate periodic labels (e.g., hour of day) from timestamps."""
    return np.array([t % period for t in timestamps])

def compute_silhouette_per_label(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute Silhouette score per label using the logic from tsne_clustering_metrics.py."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or embeddings.shape[0] < 2:
        return {str(label): 0.0 for label in unique_labels}
    
    sample_scores = silhouette_samples(embeddings, labels, metric="euclidean")
    
    results: Dict[str, float] = {}
    for label in unique_labels:
        mask = labels == label
        count = int(np.count_nonzero(mask))
        if count < 2:
            results[str(label)] = 0.0
        else:
            results[str(label)] = float(np.mean(sample_scores[mask]))
    return results
