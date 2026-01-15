import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import sys
import os
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import training_utils as tu
from evaluation_down_tasks.zeroshot_utils import build_dual_encoder_model_from_checkpoint, load_zeroshot_config

def load_dual_encoder_model(
    model_config_path: Union[str, Path],
    forecast_checkpoint_path: Optional[Union[str, Path]] = None,
    encoder_checkpoint_path: Optional[Union[str, Path]] = None,
    visual_encoder_checkpoint_path: Optional[Union[str, Path]] = None,
    requested_horizons: List[int] = [96, 192, 336, 720],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    is_meta_config: bool = False
) -> Tuple[torch.nn.Module, Dict[str, Any], List[int], int]:
    """
    Wrapper to load the dual encoder forecasting model.
    supports both direct model config and meta-config (zeroshot config).
    """
    device_obj = torch.device(device)
    model_config_path = Path(model_config_path)

    if is_meta_config:
        # Use existing utility to load meta-config which resolves relative paths
        meta_cfg = load_zeroshot_config(model_config_path)
        
        # Load the base model config referenced in meta-config
        base_model_cfg = tu.load_config(Path(meta_cfg.model_config_path))
        model_cfg = base_model_cfg.model
        
        # Apply overrides from meta-config
        if meta_cfg.overrides:
            for k, v in meta_cfg.overrides.items():
                if v is not None:
                    model_cfg[k] = v
        
        # Use paths from meta-config if not provided as arguments
        f_ckpt = Path(forecast_checkpoint_path) if forecast_checkpoint_path else Path(meta_cfg.forecast_checkpoint_path)
        e_ckpt = Path(encoder_checkpoint_path) if encoder_checkpoint_path else (Path(meta_cfg.encoder_checkpoint_path) if meta_cfg.encoder_checkpoint_path else None)
        v_ckpt = Path(visual_encoder_checkpoint_path) if visual_encoder_checkpoint_path else (Path(meta_cfg.visual_encoder_checkpoint_path) if meta_cfg.visual_encoder_checkpoint_path else None)
        
        horizons = meta_cfg.horizons if not requested_horizons else requested_horizons
    else:
        print("  [load_dual_encoder_model] Loading standard config...", flush=True)
        # Standard direct loading
        base_config = tu.load_config(model_config_path)
        model_cfg = base_config.model
        f_ckpt = Path(forecast_checkpoint_path) if forecast_checkpoint_path else None
        e_ckpt = Path(encoder_checkpoint_path) if encoder_checkpoint_path else None
        v_ckpt = Path(visual_encoder_checkpoint_path) if visual_encoder_checkpoint_path else None
        horizons = requested_horizons

    if f_ckpt is None:
        raise ValueError("Forecast checkpoint path must be provided or available in meta-config.")

    # Use existing utility to build the model
    model, checkpoint_info, eval_horizons, max_horizon, _ = build_dual_encoder_model_from_checkpoint(
        model_cfg=model_cfg,
        checkpoint_path=f_ckpt,
        requested_horizons=horizons,
        device=device_obj,
        encoder_checkpoint_path=e_ckpt,
        visual_encoder_checkpoint_path=v_ckpt,
    )
    
    return model, checkpoint_info, eval_horizons, max_horizon
