"""Project concatenated Chronos dual-encoder embeddings via t-SNE."""

from __future__ import annotations

# Removed legacy sys.path hack

from ssm_time_series import training as tu
from embeddings_visualization.projection_utils import (
    build_chronos_dataset_groups,
    collect_dual_embeddings_for_dataset,
    determine_config_path,
    load_projection_config,
    load_projection_head,
    resolve_device,
    run_tsne_projection,
    set_seed_everywhere,
)
from ssm_time_series.utils.nn import load_encoder_checkpoint

TSNE_CONFIG_ENV_VAR = "TSNE_DUAL_ENCODER_CONFIG"
DEFAULT_TSNE_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection.yaml"
DEFAULT_MODEL_CONFIG_PATH = SRC_DIR / "configs" / "mamba_encoder.yaml"
DEFAULT_OUTPUT_PREFIX = "tsne_dual_encoder_projection"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "tsne_dual_encoder"
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

    seed = proj_cfg.seed if proj_cfg.seed is not None else base_config.seed
    set_seed_everywhere(seed)

    device = resolve_device(proj_cfg.device)
    print(f"Using device: {device}")

    encoder = tu.build_encoder_from_config(base_config.model).to(device)
    visual_encoder = tu.build_visual_encoder_from_config(base_config.model).to(device)

    if proj_cfg.encoder_checkpoint is None:
        raise ValueError("Encoder checkpoint path missing in configuration")
    if proj_cfg.visual_encoder_checkpoint is None:
        raise ValueError("Visual encoder checkpoint path missing in configuration")

    print(f"Loading encoder checkpoint: {proj_cfg.encoder_checkpoint}")
    load_encoder_checkpoint(encoder, proj_cfg.encoder_checkpoint, device)
    print(f"Loading visual encoder checkpoint: {proj_cfg.visual_encoder_checkpoint}")
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

    dataset_groups = build_chronos_dataset_groups(proj_cfg=proj_cfg)

    def _collect(group) -> tuple[np.ndarray, List[int]]:
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
        title="Chronos Dual Encoder Embeddings (t-SNE)",
        checkpoint_for_timestamp=proj_cfg.encoder_checkpoint,
    )

    print("\nSaved t-SNE artifacts:")
    print(f"  Coordinates CSV: {csv_path}")
    print(f"  Plot:            {plot_path}")
    print("\nModel references:")
    print(f"  TSNE config:      {projection_config_path}")
    print(f"  Encoder config:   {proj_cfg.model_config_path}")
    print(f"  Encoder checkpoint: {proj_cfg.encoder_checkpoint}")
    print(f"  Visual checkpoint:  {proj_cfg.visual_encoder_checkpoint}")
    if proj_cfg.projection_checkpoint is not None:
        print(f"  Encoder projection head: {proj_cfg.projection_checkpoint}")
    if proj_cfg.visual_projection_checkpoint is not None:
        print(f"  Visual projection head:  {proj_cfg.visual_projection_checkpoint}")


if __name__ == "__main__":
    main()
