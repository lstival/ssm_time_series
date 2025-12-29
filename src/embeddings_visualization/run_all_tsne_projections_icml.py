"""Run temporal, visual, and dual t-SNE projections on ICML datasets.

This script mirrors:
- src/embeddings_visualization/run_all_tsne_projections.py

but swaps the dataset backend from Chronos (HF datasets) to the repository's
ICML_datasets loaders (TimeSeriesDataModule).

It reuses the same model checkpoints and projection heads as the existing
Chronos projection scripts; only the dataset source changes.

Typical usage (PowerShell):
  python src/embeddings_visualization/run_all_tsne_projections_icml.py \
    --data-dir .\\ICML_datasets \
        --dataset-names ETTm2.csv PEMS04.npz weather.csv exchange_rate.csv electricity.csv solar_AL.txt

Configs (defaults):
- Temporal: src/configs/tsne_encoder_projection_temporal.yaml
- Visual:   src/configs/tsne_encoder_projection_visual.yaml
- Dual:     src/configs/tsne_encoder_projection_dual.yaml

The YAML format is identical to the Chronos runners; this script applies
ICML-specific overrides (data_dir/dataset_names/split) at runtime.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent

for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import training_utils as tu
from dataloaders.utils import discover_dataset_files
from embeddings_visualization.projection_utils import (
    build_icml_dataset_groups,
    collect_dual_embeddings_for_dataset,
    collect_embeddings_for_dataset,
    load_projection_config,
    load_projection_head,
    resolve_device,
    run_tsne_projection,
    set_seed_everywhere,
)
from util import load_encoder_checkpoint


DEFAULT_ICML_DATASET_NAMES: List[str] = [
    "ETTm2.csv",
    "PEMS04.npz",
    "weather.csv",
    "exchange_rate.csv",
    "electricity.csv",
    "solar_AL.txt",
]

DEFAULT_TEMPORAL_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection_temporal.yaml"
DEFAULT_VISUAL_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection_visual.yaml"
DEFAULT_DUAL_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection_dual.yaml"

DEFAULT_MODEL_CONFIG_PATH = SRC_DIR / "configs" / "mamba_encoder.yaml"
DEFAULT_SPLIT = "all"


def _icml_dataset_type_caption(name: str) -> str:
    """Map ICML dataset names/paths to fixed legend captions.

    Required captions:
    - Ettm, PEMS, Weather, Exchange, Electricity, Solar
    """

    lower = (name or "").replace("\\", "/").lower()

    if "ettm" in lower:
        return "Ettm"
    if "pems" in lower:
        return "PEMS"
    if "weather" in lower:
        return "Weather"
    if "exchange" in lower:
        return "Exchange"
    if "electricity" in lower:
        return "Electricity"
    if "solar" in lower:
        return "Solar"

    # Fallback: keep the existing heuristic behavior.
    from embeddings_visualization.projection_utils import infer_dataset_type

    return infer_dataset_type(name)


def _resolve_config(path_value: Optional[str], *, default_path: Path) -> Path:
    candidate = Path(path_value).expanduser().resolve() if path_value else default_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {candidate}")


def _default_dataset_names(data_dir: Path, *, limit: int = 5) -> List[str]:
    discovered = discover_dataset_files(str(data_dir))
    existing_names = {Path(path).name for path in discovered.values()}

    preferred = [name for name in DEFAULT_ICML_DATASET_NAMES if name in existing_names]
    if preferred:
        return preferred

    names = sorted(existing_names)
    return names[: max(0, int(limit))]


def _apply_icml_overrides(
    *,
    proj_cfg,
    data_dir: Path,
    dataset_names: Optional[List[str]],
    dataset_name: Optional[str],
    split: str,
    output_prefix_suffix: str,
):
    cleaned_dataset_names: Optional[List[str]] = None
    if dataset_names:
        cleaned_dataset_names = [str(name).strip() for name in dataset_names if str(name).strip()]
        if not cleaned_dataset_names:
            cleaned_dataset_names = None

    dataset_name_value = str(dataset_name or proj_cfg.dataset_name or "").strip()

    if cleaned_dataset_names is None and not dataset_name_value:
        inferred = _default_dataset_names(data_dir, limit=5)
        if inferred:
            cleaned_dataset_names = inferred
            print(f"No datasets specified; defaulting to: {cleaned_dataset_names}")

    if cleaned_dataset_names is None and not dataset_name_value:
        raise ValueError(
            "No ICML datasets selected. Provide --dataset-names (e.g. ETTh1.csv PEMS03.npz) "
            "or --dataset-name, and ensure --data-dir points to ICML_datasets."
        )

    return replace(
        proj_cfg,
        data_dir=data_dir,
        dataset_names=cleaned_dataset_names,
        dataset_name=dataset_name_value,
        split=str(split or proj_cfg.split or DEFAULT_SPLIT).lower(),
        output_prefix=f"{proj_cfg.output_prefix}{output_prefix_suffix}",
    )


def _run_temporal(
    *,
    config_path: Path,
    data_dir: Path,
    dataset_names: Optional[List[str]],
    dataset_name: Optional[str],
    split: str,
) -> None:
    proj_cfg = load_projection_config(
        config_path,
        default_model_config_path=DEFAULT_MODEL_CONFIG_PATH,
        default_output_dir=ROOT_DIR / "results" / "tsne_encoder",
        default_output_prefix="tsne_encoder_projection",
        default_data_dir=data_dir,
        default_split=DEFAULT_SPLIT,
    )
    proj_cfg = _apply_icml_overrides(
        proj_cfg=proj_cfg,
        data_dir=data_dir,
        dataset_names=dataset_names,
        dataset_name=dataset_name,
        split=split,
        output_prefix_suffix="_icml_temporal",
    )

    base_config = tu.load_config(proj_cfg.model_config_path)
    seed = proj_cfg.seed if proj_cfg.seed is not None else base_config.seed
    set_seed_everywhere(seed)

    device = resolve_device(proj_cfg.device)
    print(f"Using device: {device}")

    encoder = tu.build_encoder_from_config(base_config.model).to(device)
    load_encoder_checkpoint(encoder, proj_cfg.encoder_checkpoint, device)
    encoder.eval()

    projection_head = load_projection_head(
        encoder=encoder,
        projection_checkpoint_path=proj_cfg.projection_checkpoint,
        device=device,
    )

    dataset_groups = build_icml_dataset_groups(proj_cfg=proj_cfg)

    def _collect(group):
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
        title="ICML Temporal Encoder Embeddings (t-SNE)",
        checkpoint_for_timestamp=proj_cfg.encoder_checkpoint,
        dataset_type_fn=_icml_dataset_type_caption,
    )

    print("\nSaved temporal t-SNE artifacts:")
    print(f"  Coordinates CSV: {csv_path}")
    print(f"  Plot:            {plot_path}")


def _run_visual(
    *,
    config_path: Path,
    data_dir: Path,
    dataset_names: Optional[List[str]],
    dataset_name: Optional[str],
    split: str,
) -> None:
    proj_cfg = load_projection_config(
        config_path,
        default_model_config_path=DEFAULT_MODEL_CONFIG_PATH,
        default_output_dir=ROOT_DIR / "results" / "tsne_visual_encoder",
        default_output_prefix="tsne_visual_encoder_projection",
        default_data_dir=data_dir,
        default_split=DEFAULT_SPLIT,
    )
    proj_cfg = _apply_icml_overrides(
        proj_cfg=proj_cfg,
        data_dir=data_dir,
        dataset_names=dataset_names,
        dataset_name=dataset_name,
        split=split,
        output_prefix_suffix="_icml_visual",
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

    dataset_groups = build_icml_dataset_groups(proj_cfg=proj_cfg)

    def _collect(group):
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
        title="ICML Visual Encoder Embeddings (t-SNE)",
        checkpoint_for_timestamp=visual_encoder_checkpoint,
        dataset_type_fn=_icml_dataset_type_caption,
    )

    print("\nSaved visual t-SNE artifacts:")
    print(f"  Coordinates CSV: {csv_path}")
    print(f"  Plot:            {plot_path}")


def _run_dual(
    *,
    config_path: Path,
    data_dir: Path,
    dataset_names: Optional[List[str]],
    dataset_name: Optional[str],
    split: str,
) -> None:
    proj_cfg = load_projection_config(
        config_path,
        default_model_config_path=DEFAULT_MODEL_CONFIG_PATH,
        default_output_dir=ROOT_DIR / "results" / "tsne_dual_encoder",
        default_output_prefix="tsne_dual_encoder_projection",
        default_data_dir=data_dir,
        default_split=DEFAULT_SPLIT,
    )
    proj_cfg = _apply_icml_overrides(
        proj_cfg=proj_cfg,
        data_dir=data_dir,
        dataset_names=dataset_names,
        dataset_name=dataset_name,
        split=split,
        output_prefix_suffix="_icml_dual",
    )

    base_config = tu.load_config(proj_cfg.model_config_path)

    seed = proj_cfg.seed if proj_cfg.seed is not None else base_config.seed
    set_seed_everywhere(seed)

    device = resolve_device(proj_cfg.device)
    print(f"Using device: {device}")

    encoder = tu.build_encoder_from_config(base_config.model).to(device)
    visual_encoder = tu.build_visual_encoder_from_config(base_config.model).to(device)

    if proj_cfg.visual_encoder_checkpoint is None:
        raise ValueError("Visual encoder checkpoint path missing in configuration")

    load_encoder_checkpoint(encoder, proj_cfg.encoder_checkpoint, device)
    load_encoder_checkpoint(visual_encoder, proj_cfg.visual_encoder_checkpoint, device)

    encoder.eval()
    visual_encoder.eval()

    encoder_projection_head = load_projection_head(
        encoder=encoder,
        projection_checkpoint_path=proj_cfg.projection_checkpoint,
        device=device,
    )
    visual_projection_head = load_projection_head(
        encoder=visual_encoder,
        projection_checkpoint_path=proj_cfg.visual_projection_checkpoint,
        device=device,
    )

    dataset_groups = build_icml_dataset_groups(proj_cfg=proj_cfg)

    def _collect(group):
        return collect_dual_embeddings_for_dataset(
            encoder=encoder,
            visual_encoder=visual_encoder,
            encoder_proj=encoder_projection_head,
            visual_proj=visual_projection_head,
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
        title="ICML Dual Encoder Embeddings (t-SNE)",
        checkpoint_for_timestamp=proj_cfg.encoder_checkpoint,
        dataset_type_fn=_icml_dataset_type_caption,
    )

    print("\nSaved dual t-SNE artifacts:")
    print(f"  Coordinates CSV: {csv_path}")
    print(f"  Plot:            {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run temporal, visual, and dual t-SNE projections on ICML datasets (same models/checkpoints)."
    )
    parser.add_argument(
        "--temporal-config",
        type=str,
        default=None,
        help="Path to YAML config for temporal encoder projection.",
    )
    parser.add_argument(
        "--visual-config",
        type=str,
        default=None,
        help="Path to YAML config for visual encoder projection.",
    )
    parser.add_argument(
        "--dual-config",
        type=str,
        default=None,
        help="Path to YAML config for dual (temporal+visual) projection.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str((ROOT_DIR / "ICML_datasets").resolve()),
        help="Path to ICML_datasets root directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Single dataset filter (e.g. ETTh1.csv).",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="*",
        default=None,
        help="Dataset filenames to include (e.g. ETTh1.csv ETTm1.csv PEMS03.npz).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to use from ICML loaders.",
    )

    args = parser.parse_args()

    temporal_cfg = _resolve_config(args.temporal_config, default_path=DEFAULT_TEMPORAL_CONFIG_PATH)
    visual_cfg = _resolve_config(args.visual_config, default_path=DEFAULT_VISUAL_CONFIG_PATH)
    dual_cfg = _resolve_config(args.dual_config, default_path=DEFAULT_DUAL_CONFIG_PATH)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"ICML data directory not found: {data_dir}")

    print("\n=== Temporal projection (ICML) ===")
    print(f"Config: {temporal_cfg}")
    _run_temporal(
        config_path=temporal_cfg,
        data_dir=data_dir,
        dataset_names=args.dataset_names,
        dataset_name=args.dataset_name,
        split=args.split,
    )

    print("\n=== Visual projection (ICML) ===")
    print(f"Config: {visual_cfg}")
    _run_visual(
        config_path=visual_cfg,
        data_dir=data_dir,
        dataset_names=args.dataset_names,
        dataset_name=args.dataset_name,
        split=args.split,
    )

    print("\n=== Dual projection (ICML) ===")
    print(f"Config: {dual_cfg}")
    _run_dual(
        config_path=dual_cfg,
        data_dir=data_dir,
        dataset_names=args.dataset_names,
        dataset_name=args.dataset_name,
        split=args.split,
    )


if __name__ == "__main__":
    main()
