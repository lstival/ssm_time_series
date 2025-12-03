"""Project concatenated embeddings from frozen dual encoders via t-SNE."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from evaluation_down_tasks.zeroshot_utils import select_loader
from embeddings_visualization.projection_utils import (
    build_output_dir,
    determine_config_path,
    fit_tsne,
    infer_dataset_type,
    load_projection_config,
    load_projection_head,
)
from time_series_loader import TimeSeriesDataModule
from util import default_device, load_encoder_checkpoint

TSNE_CONFIG_ENV_VAR = "TSNE_DUAL_ENCODER_CONFIG"
DEFAULT_TSNE_CONFIG_PATH = SRC_DIR / "configs" / "tsne_encoder_projection.yaml"
DEFAULT_MODEL_CONFIG_PATH = SRC_DIR / "configs" / "mamba_encoder.yaml"
DEFAULT_OUTPUT_PREFIX = "tsne_dual_encoder_projection"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "tsne_dual_encoder"
DEFAULT_DATA_DIR = ROOT_DIR / "ICML_datasets"
DEFAULT_SPLIT = "all"

def _collect_dual_embeddings(
    *,
    encoder: torch.nn.Module,
    visual_encoder: torch.nn.Module,
    encoder_proj: Optional[torch.nn.Module],
    visual_proj: Optional[torch.nn.Module],
    loader,
    device: torch.device,
    samples_per_dataset: int,
    sequence_first_input: bool,
) -> Tuple[np.ndarray, List[int]]:
    if loader is None:
        return np.empty((0, 0), dtype=np.float32), []

    encoder.eval()
    visual_encoder.eval()
    collected: List[torch.Tensor] = []
    sample_indices: List[int] = []
    local_index = 0

    with torch.no_grad():
        for batch in loader:
            seq_x = batch[0].float().to(device)
            if not sequence_first_input:
                seq_x = seq_x.transpose(1, 2)

            encoder_embed = encoder(seq_x)
            visual_embed = visual_encoder(seq_x)

            if encoder_proj is not None:
                encoder_embed = encoder_proj(encoder_embed)
            if visual_proj is not None:
                visual_embed = visual_proj(visual_embed)

            if encoder_embed.dim() > 2:
                encoder_embed = encoder_embed.flatten(start_dim=1)
            if visual_embed.dim() > 2:
                visual_embed = visual_embed.flatten(start_dim=1)

            combined = torch.cat([encoder_embed, visual_embed], dim=1).detach().cpu()
            collected.append(combined)

            batch_size = combined.size(0)
            sample_indices.extend(range(local_index, local_index + batch_size))
            local_index += batch_size

            if local_index >= samples_per_dataset:
                break

    if not collected:
        return np.empty((0, 0), dtype=np.float32), []

    stacked = torch.cat(collected, dim=0)
    if stacked.size(0) > samples_per_dataset:
        stacked = stacked[:samples_per_dataset]
        sample_indices = sample_indices[:samples_per_dataset]

    return stacked.numpy(), sample_indices


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
    if seed is not None:
        tu.set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    device_spec = proj_cfg.device.strip().lower()
    if device_spec == "auto":
        device = default_device()
    else:
        device = torch.device(proj_cfg.device)
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

    module = TimeSeriesDataModule(
        dataset_name=proj_cfg.dataset_name,
        dataset_names=proj_cfg.dataset_names,
        data_dir=str(proj_cfg.data_dir),
        batch_size=proj_cfg.batch_size,
        val_batch_size=proj_cfg.val_batch_size,
        num_workers=proj_cfg.num_workers,
        pin_memory=True,
        normalize=True,
        filename=str(proj_cfg.filename) if proj_cfg.filename is not None else None,
        train=True,
        val=True,
        test=True,
    )

    dataset_groups = module.get_dataloaders()
    if not dataset_groups:
        raise RuntimeError("No datasets available for visualization")

    dataset_entries: List[Dict[str, object]] = []

    for group in dataset_groups:
        loader, _dataset_obj, split_used = select_loader(group, proj_cfg.split)
        if loader is None:
            print(f"Skipping dataset '{group.name}' because split '{proj_cfg.split}' is unavailable.")
            continue

        dataset_label = group.name
        dataset_type = infer_dataset_type(dataset_label)

        try:
            embeddings, sample_indices = _collect_dual_embeddings(
                encoder=encoder,
                visual_encoder=visual_encoder,
                encoder_proj=encoder_projection_head,
                visual_proj=visual_projection_head,
                loader=loader,
                device=device,
                samples_per_dataset=proj_cfg.samples_per_dataset,
                sequence_first_input=proj_cfg.sequence_first_input,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"  Skipping dataset '{dataset_label}' due to runtime error: {exc}")
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            continue

        sample_total = embeddings.shape[0]
        if sample_total == 0:
            print(f"  Dataset '{dataset_label}' produced no samples; skipping.")
            continue

        dataset_entries.append(
            {
                "name": dataset_label,
                "type": dataset_type,
                "split": split_used,
                "embeddings": embeddings,
                "indices": sample_indices,
            }
        )
        print(f"  Collected {sample_total} dual embeddings for '{dataset_label}' ({dataset_type}).")

    if not dataset_entries:
        raise RuntimeError("No embeddings collected from the requested datasets.")

    embedding_blocks = [np.asarray(entry["embeddings"], dtype=np.float32) for entry in dataset_entries]
    embedding_matrix = np.concatenate(embedding_blocks, axis=0)

    coords = fit_tsne(
        embedding_matrix,
        perplexity=proj_cfg.perplexity,
        learning_rate=proj_cfg.learning_rate,
        n_iter=proj_cfg.n_iter,
        seed=seed,
    )

    records: List[Dict[str, object]] = []
    for entry in dataset_entries:
        embeddings = np.asarray(entry["embeddings"], dtype=np.float32)
        count = embeddings.shape[0]
        indices = entry["indices"]
        for idx in range(count):
            records.append(
                {
                    "dataset": entry["name"],
                    "dataset_type": entry["type"],
                    "split": entry["split"],
                    "sample_in_dataset": indices[idx] if idx < len(indices) else idx,
                }
            )

    coords_df = pd.DataFrame.from_records(records)
    coords_df["tsne_x"] = coords[:, 0]
    coords_df["tsne_y"] = coords[:, 1]

    results_dir = proj_cfg.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = build_output_dir(results_dir, proj_cfg.output_prefix, proj_cfg.encoder_checkpoint)
    csv_path = output_dir / "tsne_coordinates.csv"
    plot_path = output_dir / "tsne_plot.png"

    coords_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_types = sorted(coords_df["dataset_type"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_types))))

    for color, dataset_type in zip(colors, unique_types):
        mask = coords_df["dataset_type"] == dataset_type
        ax.scatter(
            coords_df.loc[mask, "tsne_x"],
            coords_df.loc[mask, "tsne_y"],
            label=dataset_type,
            s=16,
            alpha=0.75,
            color=color,
        )

    ax.set_title("ICML Dual Encoder Embeddings (t-SNE)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)

    if proj_cfg.show:
        plt.show()
    plt.close(fig)

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
