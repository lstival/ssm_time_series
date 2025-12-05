"""Utility functions for Comet ML experiment tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
import comet_ml


def load_comet_config(config_path: Optional[Path] = None) -> dict:
    """Load Comet ML configuration from YAML file.
    
    Args:
        config_path: Path to comet_config.yaml. If None, uses default location.
        
    Returns:
        Dictionary containing Comet ML configuration.
    """
    if config_path is None:
        # Default to configs/comet_config.yaml relative to this file
        src_dir = Path(__file__).resolve().parent
        config_path = src_dir.parent / "comet_configs" / "comet_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Comet config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config.get("comet", {})


def create_comet_experiment(
    experiment_name_key: str,
    config_path: Optional[Path] = None,
    **kwargs
) -> comet_ml.Experiment:
    """Create a Comet ML experiment with configuration from YAML file.
    
    Args:
        experiment_name_key: Key for experiment name in config (e.g., 'chronos_supervised')
        config_path: Path to comet_config.yaml. If None, uses default location.
        **kwargs: Additional arguments to pass to comet_ml.Experiment
        
    Returns:
        Initialized Comet ML Experiment object.
    """
    comet_config = load_comet_config(config_path)
    
    # Get experiment name from config
    experiment_names = comet_config.get("experiment_names", {})
    experiment_name = experiment_names.get(experiment_name_key, experiment_name_key)
    
    # Create experiment
    experiment = comet_ml.Experiment(
        api_key=comet_config.get("api_key"),
        project_name=comet_config.get("project_name"),
        workspace=comet_config.get("workspace"),
        **kwargs
    )
    
    experiment.set_name(experiment_name)
    
    return experiment
