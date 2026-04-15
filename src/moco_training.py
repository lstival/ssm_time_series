"""Compatibility helpers for training entrypoints.

This module was historically imported by multiple scripts for path and
checkpoint resolution utilities.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional


def resolve_path(base: Path, candidate: Optional[object]) -> Optional[Path]:
    """Resolve a path candidate against a base directory.

    Returns None when candidate is missing or the resolved path does not exist.
    """
    if candidate is None:
        return None
    path = Path(candidate)
    if path.is_absolute():
        return path if path.exists() else None
    resolved = (base / path).resolve()
    return resolved if resolved.exists() else None


def resolve_checkpoint_dir(config, config_path: Path, override: Optional[Path]) -> Path:
    """Build checkpoint output directory from config or explicit override."""
    if override is not None:
        return Path(override).resolve()

    log_cfg = config.logging
    ckpt_rel = log_cfg.get("checkpoint_dir", "../../checkpoints")
    base = (config_path.parent / ckpt_rel).resolve()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = str(getattr(config, "experiment_name", "experiment"))
    return base / f"{experiment_name}_{timestamp}"
