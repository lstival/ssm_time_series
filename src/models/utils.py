from pyts.image import RecurrencePlot
import numpy as np
import torch
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Iterable
from dataclasses import dataclass
import datasets

def time_series_2_recurrence_plot(x):
    # normalize input to numpy array (handle torch tensors if provided)
    try:
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
    except Exception:
        arr = np.asarray(x)

    # Supported shapes:
    # (n_samples, length), (length,), (n_samples, n_channels, length)
    if arr.ndim == 1:
        arr2 = arr[None, :]
        imgs = RecurrencePlot().fit_transform(arr2)  # (1, L, L)
        return imgs.astype(np.float32)
    elif arr.ndim == 2:
        arr2 = arr
        imgs = RecurrencePlot().fit_transform(arr2)  # (n_samples, L, L)
        return imgs.astype(np.float32)
    elif arr.ndim == 3:
        n_samples, n_channels, length = arr.shape
        # If single channel, behave as before and return (n_samples, length, length)
        if n_channels == 1:
            arr2 = arr[:, 0, :]
            imgs = RecurrencePlot().fit_transform(arr2)  # (n_samples, L, L)
            return imgs.astype(np.float32)
        # For multichannel, compute one RP per channel per sample -> (n_samples, n_channels, L, L)
        arr2 = arr.reshape(n_samples * n_channels, length)
        imgs = RecurrencePlot().fit_transform(arr2)  # (n_samples*n_channels, L, L)
        imgs = imgs.reshape(n_samples, n_channels, length, length)
        return imgs.astype(np.float32)
    else:
        raise ValueError(f"Unsupported input shape: {arr.shape}")




@dataclass
class ChronosForecastConfig:
    """Configuration class for Chronos forecasting."""
    # Model configuration
    model_config_path: Path
    overrides: Dict[str, object]
    
    # Training configuration
    horizons: List[int]
    max_horizon: int
    seed: int
    batch_size: int
    val_batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    mlp_hidden_dim: int
    num_workers: int
    pin_memory: bool
    
    # Chronos dataset configuration
    datasets_to_load: List[str]
    split: str
    repo_id: str
    target_dtype: Optional[str]
    normalize_per_series: bool
    val_ratio: float
    context_length: int
    stride: int
    max_windows_per_series: Optional[int]
    max_series: Optional[int]
    load_kwargs: Dict[str, object]
    
    # Path configuration
    visual_mamba_checkpoint_path: Path
    checkpoint_dir: Path
    results_dir: Path
    
    # Configuration metadata
    config_path: Path
    config_dir: Path


def _parse_dataset_names(raw: Optional[Iterable[str] | str]) -> List[str]:
    """Parse dataset names from various input formats."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [entry.strip() for entry in raw.split(",") if entry.strip()]
    result: List[str] = []
    for entry in raw:
        if entry is None:
            continue
        name = str(entry).strip()
        if name:
            result.append(name)
    return result


def _load_yaml_config(path: Path) -> Dict[str, object]:
    """Load YAML configuration from file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must define a mapping: {path}")
    return payload


def _resolve_optional_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    """Resolve optional path relative to base directory."""
    if candidate is None:
        return None
    candidate_path = Path(str(candidate))
    # Try to resolve relative to base first
    if not candidate_path.is_absolute():
        resolved = (base / candidate_path).resolve()
        if resolved.exists():
            return resolved
    return candidate_path.expanduser().resolve()


def _coerce_path(base: Path, candidate: object, *, must_exist: bool = False, description: str) -> Path:
    """Coerce and validate path."""
    resolved = _resolve_optional_path(base, candidate)
    if resolved is None:
        raise FileNotFoundError(f"{description} not found: {candidate}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _normalize_horizons(raw: object) -> List[int]:
    """Normalize horizon values from config."""
    if raw is None:
        raise ValueError("Configuration must supply 'training.horizons'.")
    if isinstance(raw, str):
        # Assume comma-separated string
        try:
            values = [int(v.strip()) for v in raw.split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError("'training.horizons' string must be comma-separated integers.") from exc
    else:
        try:
            values = [int(value) for value in raw]  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("'training.horizons' must be a list of integers or a comma-separated string.") from exc
    return sorted(set(values))


def _ensure_hf_list_feature_registered() -> None:
    """Ensure HuggingFace List feature is registered."""
    feature_registry = getattr(datasets.features, "_FEATURE_TYPES", None)
    sequence_cls = getattr(datasets.features, "Sequence", None)
    if isinstance(feature_registry, dict) and "List" not in feature_registry and sequence_cls is not None:
        feature_registry["List"] = sequence_cls


def load_chronos_forecast_config(
    config_path: Optional[Path] = None,
    env_var: str = "CHRONOS_FORECAST_CONFIG",
    default_config_name: str = "chronos_forecast.yaml"
) -> ChronosForecastConfig:
    """Load and parse Chronos forecast configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        env_var: Environment variable name for config override
        default_config_name: Default config filename to look for
        
    Returns:
        ChronosForecastConfig: Parsed configuration object
    """
    # Determine config path
    if config_path is None:
        env_override = os.getenv(env_var)
        if env_override:
            config_path = Path(env_override)
        else:
            # Look for default config in src/configs/
            src_dir = Path(__file__).resolve().parents[1]  # Go up from models to src
            config_path = src_dir / "configs" / default_config_name
    
    config_path = _resolve_optional_path(Path.cwd(), config_path)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Chronos forecast configuration not found: {config_path}")
    
    print(f"Using forecast configuration: {config_path}")
    
    forecast_cfg = _load_yaml_config(config_path)
    config_dir = config_path.parent
    
    # Load model configuration
    model_section = dict(forecast_cfg.get("model") or {})
    model_config_candidate = model_section.get("config")
    if model_config_candidate is None:
        raise ValueError("Configuration missing required key 'model.config'.")
    model_config_path = _coerce_path(
        config_dir,
        model_config_candidate,
        must_exist=True,
        description="Model configuration",
    )
    
    # Import training_utils to load base config
    import sys
    src_dir = Path(__file__).resolve().parents[1]
    root_dir = src_dir.parent
    for path in (src_dir, root_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    import training_utils as tu
    base_config = tu.load_config(model_config_path)
    
    data_cfg = dict(base_config.data or {})
    logging_cfg = dict(base_config.logging or {})
    
    overrides = dict(model_section.get("overrides") or {})
    
    # Parse training configuration
    training_section = dict(forecast_cfg.get("training") or {})
    horizons = _normalize_horizons(training_section.get("horizons"))
    max_horizon = max(horizons)
    
    seed = int(training_section.get("seed", base_config.seed))
    batch_size = int(training_section.get("batch_size", data_cfg.get("batch_size", 16)))
    val_batch_size = int(training_section.get("val_batch_size", batch_size))
    epochs = int(training_section.get("epochs", 2))
    lr = float(training_section.get("lr", 3e-4))
    weight_decay = float(training_section.get("weight_decay", 1e-2))
    mlp_hidden_dim = int(training_section.get("mlp_hidden_dim", 512))
    num_workers = int(training_section.get("num_workers", data_cfg.get("num_workers", 0)))
    pin_memory = bool(training_section.get("pin_memory", data_cfg.get("pin_memory", True)))
    
    # Parse Chronos configuration
    chronos_section = dict(forecast_cfg.get("chronos") or {})
    loader_config_candidate = chronos_section.get("config")
    chronos_cfg: Dict[str, object] = {}
    if loader_config_candidate is not None:
        loader_config_path = _coerce_path(
            config_dir,
            loader_config_candidate,
            must_exist=True,
            description="Chronos loader configuration",
        )
        with loader_config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Chronos loader configuration must be a mapping: {loader_config_path}")
        chronos_cfg.update(payload)
    
    # Parse dataset configuration
    datasets_value = chronos_section.get("datasets")
    if datasets_value is None:
        datasets_value = chronos_cfg.get("datasets_to_load") or chronos_cfg.get("datasets")
    datasets_to_load = _parse_dataset_names(datasets_value)
    if not datasets_to_load:
        raise ValueError("Chronos configuration must provide at least one dataset name.")
    
    split = str(chronos_section.get("split", chronos_cfg.get("split", "train")))
    repo_id = str(chronos_section.get("repo_id", chronos_cfg.get("repo_id", "autogluon/chronos_datasets")))
    target_dtype = chronos_section.get("target_dtype", chronos_cfg.get("target_dtype"))
    
    normalize_per_series = bool(chronos_section.get("normalize", chronos_cfg.get("normalize", True)))
    val_ratio = float(chronos_section.get("val_split", chronos_cfg.get("val_split", 0.2)))
    
    context_length = chronos_section.get("context_length")
    if context_length is None:
        context_length = chronos_cfg.get("context_length", chronos_cfg.get("patch_length", 384))
    context_length = int(context_length)
    
    stride_value = chronos_section.get("window_stride", chronos_cfg.get("window_stride"))
    stride = int(stride_value) if stride_value is not None else max_horizon
    stride = max(1, stride)
    
    max_windows_per_series = chronos_section.get("max_windows_per_series", chronos_cfg.get("max_windows_per_series"))
    if max_windows_per_series is not None:
        max_windows_per_series = int(max_windows_per_series)
    
    max_series = chronos_section.get("max_series", chronos_cfg.get("max_series"))
    if max_series is not None:
        max_series = int(max_series)
    
    load_kwargs: Dict[str, object] = dict(chronos_cfg.get("load_kwargs", {}) or {})
    section_load_kwargs = chronos_section.get("load_kwargs")
    if isinstance(section_load_kwargs, dict):
        load_kwargs.update(section_load_kwargs)
    
    force_offline = chronos_section.get("force_offline")
    if force_offline is not None:
        load_kwargs["force_offline"] = bool(force_offline)
    
    offline_cache_dir = load_kwargs.get("offline_cache_dir")
    if offline_cache_dir is not None:
        load_kwargs["offline_cache_dir"] = str(
            _resolve_optional_path(config_dir, offline_cache_dir)
        )
    if "offline_cache_dir" not in load_kwargs:
        load_kwargs["offline_cache_dir"] = str((root_dir / "data").resolve())
    
    # Parse paths configuration
    paths_section = dict(forecast_cfg.get("paths") or {})
    visual_candidate = paths_section.get(
        "visual_encoder_checkpoint",
        Path("../../checkpoints/ts_encoder_20251101_1100/visual_encoder_best.pt"),
    )
    visual_mamba_checkpoint_path = _coerce_path(
        config_dir,
        visual_candidate,
        must_exist=True,
        description="Visual Mamba encoder checkpoint",
    )
    
    checkpoint_candidate = paths_section.get("checkpoint_dir")
    if checkpoint_candidate is not None:
        checkpoint_base = config_dir
    else:
        checkpoint_candidate = logging_cfg.get("checkpoint_dir", root_dir / "checkpoints")
        checkpoint_base = root_dir
    checkpoint_dir = _coerce_path(
        checkpoint_base,
        checkpoint_candidate,
        must_exist=False,
        description="Checkpoint directory",
    )
    
    results_candidate = paths_section.get("results_dir")
    if results_candidate is not None:
        results_base = config_dir
    else:
        results_candidate = root_dir / "results"
        results_base = root_dir
    results_dir = _coerce_path(
        results_base,
        results_candidate,
        must_exist=False,
        description="Results directory",
    )
    
    # Ensure HF feature is registered
    _ensure_hf_list_feature_registered()
    
    return ChronosForecastConfig(
        model_config_path=model_config_path,
        overrides=overrides,
        horizons=horizons,
        max_horizon=max_horizon,
        seed=seed,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        mlp_hidden_dim=mlp_hidden_dim,
        num_workers=num_workers,
        pin_memory=pin_memory,
        datasets_to_load=datasets_to_load,
        split=split,
        repo_id=repo_id,
        target_dtype=target_dtype,
        normalize_per_series=normalize_per_series,
        val_ratio=val_ratio,
        context_length=context_length,
        stride=stride,
        max_windows_per_series=max_windows_per_series,
        max_series=max_series,
        load_kwargs=load_kwargs,
        visual_mamba_checkpoint_path=visual_mamba_checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        config_path=config_path,
        config_dir=config_dir,
    )





if __name__ == "__main__":
    x = torch.randn(4,5,32)
    out = time_series_2_recurrence_plot(x)
    print(out.shape)
    import matplotlib.pyplot as plt

    plt.imshow(out[0][:1].swapaxes(0,2), cmap="gray")