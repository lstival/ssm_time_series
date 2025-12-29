"""Run temporal, visual, and dual t-SNE projections sequentially.

This is a thin orchestrator around:
- tsne_encoder_projection.py
- tsne_visual_encoder_projection.py
- tsne_dual_encoder_projection.py

It sets the expected environment variables so each script picks up its config.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DEFAULT_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection.yaml"
DEFAULT_TEMPORAL_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection_temporal.yaml"
DEFAULT_VISUAL_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection_visual.yaml"
DEFAULT_DUAL_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection_dual.yaml"


def _resolve_config(path_value: str | None, *, default_path: Path) -> Path:
    candidate = Path(path_value).expanduser().resolve() if path_value else default_path
    if candidate.exists():
        return candidate
    if default_path != DEFAULT_CONFIG_PATH and DEFAULT_CONFIG_PATH.exists():
        print(f"Warning: {candidate} not found; falling back to {DEFAULT_CONFIG_PATH}")
        return DEFAULT_CONFIG_PATH
    raise FileNotFoundError(f"Config not found: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run temporal, visual, and dual t-SNE projections (Chronos datasets)."
    )
    parser.add_argument(
        "--temporal-config",
        type=str,
        default=None,
        help="Path to YAML config for temporal encoder projection (TSNE_ENCODER_CONFIG).",
    )
    parser.add_argument(
        "--visual-config",
        type=str,
        default=None,
        help="Path to YAML config for visual encoder projection (TSNE_VISUAL_ENCODER_CONFIG).",
    )
    parser.add_argument(
        "--dual-config",
        type=str,
        default=None,
        help="Path to YAML config for dual (temporal+visual) projection (TSNE_DUAL_ENCODER_CONFIG).",
    )

    args = parser.parse_args()

    temporal_cfg = _resolve_config(args.temporal_config, default_path=DEFAULT_TEMPORAL_CONFIG_PATH)
    visual_cfg = _resolve_config(args.visual_config, default_path=DEFAULT_VISUAL_CONFIG_PATH)
    dual_cfg = _resolve_config(args.dual_config, default_path=DEFAULT_DUAL_CONFIG_PATH)

    os.environ["TSNE_ENCODER_CONFIG"] = str(temporal_cfg)
    os.environ["TSNE_VISUAL_ENCODER_CONFIG"] = str(visual_cfg)
    os.environ["TSNE_DUAL_ENCODER_CONFIG"] = str(dual_cfg)

    from embeddings_visualization import tsne_dual_encoder_projection
    from embeddings_visualization import tsne_encoder_projection
    from embeddings_visualization import tsne_visual_encoder_projection

    print("\n=== Temporal projection ===")
    print(f"Config: {temporal_cfg}")
    tsne_encoder_projection.main()

    print("\n=== Visual projection ===")
    print(f"Config: {visual_cfg}")
    tsne_visual_encoder_projection.main()

    print("\n=== Dual projection ===")
    print(f"Config: {dual_cfg}")
    tsne_dual_encoder_projection.main()


if __name__ == "__main__":
    main()
