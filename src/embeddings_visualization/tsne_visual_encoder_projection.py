"""Project Chronos visual encoder embeddings via t-SNE with dataset-type coloring."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from embeddings_visualization.projection_utils import (
    build_chronos_dataset_groups,
    collect_embeddings_for_dataset,
    determine_config_path,
    load_projection_config,
    load_projection_head,
    resolve_device,
    run_tsne_projection,
    set_seed_everywhere,
)
from util import load_encoder_checkpoint

TSNE_CONFIG_ENV_VAR = "TSNE_VISUAL_ENCODER_CONFIG"
DEFAULT_TSNE_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection.yaml"
DEFAULT_MODEL_CONFIG_PATH = SRC_DIR / "configs" / "mamba_encoder.yaml"
DEFAULT_OUTPUT_PREFIX = "tsne_visual_encoder_projection"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "tsne_visual_encoder"
DEFAULT_DATA_DIR = ROOT_DIR / "chronos"
DEFAULT_SPLIT = "all"



def main() -> None:
    projection_config_path = determine_config_path(
        env_var=TSNE_CONFIG_ENV_VAR,
        default_path=DEFAULT_TSNE_CONFIG_PATH,
    )
    proj_cfg = load_projection_config(
        projection_config_path,
        default_model_config_path=DEFAULT_MODEL_CONFIG_PATH,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        default_output_prefix=DEFAULT_OUTPUT_PREFIX,
        default_data_dir=DEFAULT_DATA_DIR,
        default_split=DEFAULT_SPLIT,
    )
    base_config = tu.load_config(proj_cfg.model_config_path)

    visual_encoder_checkpoint = proj_cfg.visual_encoder_checkpoint or proj_cfg.encoder_checkpoint
    visual_projection_checkpoint = proj_cfg.visual_projection_checkpoint or proj_cfg.projection_checkpoint
    if visual_encoder_checkpoint is None:
        raise ValueError("Visual encoder checkpoint path is required for visual projection")

    seed = proj_cfg.seed if proj_cfg.seed is not None else base_config.seed
    set_seed_everywhere(seed)

    device = resolve_device(proj_cfg.device)
    print(f"Using device: {device}")

    encoder = tu.build_visual_encoder_from_config(base_config.model).to(device)
    load_encoder_checkpoint(encoder, visual_encoder_checkpoint, device)
    encoder.eval()

    projection_head = load_projection_head(
        encoder=encoder,
        projection_checkpoint_path=visual_projection_checkpoint,
        device=device,
    )

    dataset_groups = build_chronos_dataset_groups(proj_cfg=proj_cfg)

    def _collect(group) -> tuple[np.ndarray, List[int]]:
        return collect_embeddings_for_dataset(
            encoder=encoder,
            projection_head=projection_head,
            loader=group.loader,
            device=device,
            samples_per_dataset=proj_cfg.samples_per_dataset,
            sequence_first_input=proj_cfg.sequence_first_input,
            max_sequence_length=proj_cfg.max_sequence_length,
        )

    csv_path, plot_path = run_tsne_projection(
        dataset_groups=dataset_groups,
        collect_fn=_collect,
        proj_cfg=proj_cfg,
        seed=seed,
        title="Chronos Visual Encoder Embeddings (t-SNE)",
        checkpoint_for_timestamp=visual_encoder_checkpoint,
    )

    print("\nSaved t-SNE artifacts:")
    print(f"  Coordinates CSV: {csv_path}")
    print(f"  Plot:            {plot_path}")
    print("\nModel references:")
    print(f"  TSNE config:      {projection_config_path}")
    print(f"  Encoder config:   {proj_cfg.model_config_path}")
    print(f"  Visual checkpoint: {visual_encoder_checkpoint}")
    if visual_projection_checkpoint is not None:
        print(f"  Projection head:  {visual_projection_checkpoint}")


if __name__ == "__main__":
    main()
