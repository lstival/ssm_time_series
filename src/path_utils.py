"""Shared path helpers (replaces deleted moco_training.resolve_path / resolve_checkpoint_dir)."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional


def resolve_path(base: Path, candidate) -> Optional[Path]:
    if candidate is None:
        return None
    p = Path(candidate)
    if p.is_absolute():
        return p if p.exists() else None
    resolved = (base / p).resolve()
    return resolved if resolved.exists() else None


def resolve_checkpoint_dir(config, config_path: Path, override) -> Path:
    if override is not None:
        return Path(override).resolve()
    log_cfg = config.logging
    ckpt_rel = log_cfg.get("checkpoint_dir", "../../checkpoints")
    base = (config_path.parent / ckpt_rel).resolve()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = str(getattr(config, "experiment_name", "experiment"))
    return base / f"{exp_name}_{ts}"
