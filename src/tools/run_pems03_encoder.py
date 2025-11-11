"""Simple script to load a pretrained encoder and run it on ICML PEMS03.

Steps:
- load the pretrained time-series encoder from a checkpoint
- load the ICML PEMS03.npz dataset using the repo's current dataloaders
- iterate over the requested split and forward each batch through the encoder

This is intentionally minimal: it doesn't save outputs, it just prints shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from types import SimpleNamespace

import torch
from torch import nn

import training_utils as tu
from moco_training import resolve_path
from time_series_loader import TimeSeriesDataModule
from util import default_device
import numpy as np


def _load_encoder(config: tu.ExperimentConfig, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model_cfg = dict(config.model or {})
    encoder = tu.build_encoder_from_config(model_cfg).to(device)
    encoder.eval()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint format at {checkpoint_path}")

    candidates = (
        payload.get("model_state_dict"),
        payload.get("encoder_state_dict"),
        payload.get("state_dict"),
        payload.get("encoder"),
        payload.get("model"),
    )
    state_dict = next((item for item in candidates if isinstance(item, dict)), None)
    state_dict = state_dict or payload

    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing encoder weights: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected encoder weights: {sorted(unexpected)}")
    return encoder


if __name__ == "__main__":  # pragma: no cover - simple CLI entrypoint
    args = SimpleNamespace(
        config=str(Path(__file__).resolve().parents[1] / "configs" / "mamba_encoder.yaml"),
        encoder_checkpoint=str(Path(__file__).resolve().parents[2] / "checkpoints" / "ts_encoder_20251101_1100" / "time_series_best.pt"),
        data_dir=None,
        split="test",
        device="auto",
        max_batches=5,
    )

    cfg_path = resolve_path(Path.cwd(), Path(args.config))
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(cfg_path)

    device = default_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    # Resolve data directory (prefer CLI, then config, all relative to config)
    data_cfg = dict(config.data or {})
    base_dir = cfg_path.parent
    default_data_dir = data_cfg.get("data_dir", "../../ICML_datasets")
    data_dir_input = args.data_dir or default_data_dir
    data_dir_path = resolve_path(base_dir, data_dir_input)
    data_dir_final = str(data_dir_path) if data_dir_path is not None else str(data_dir_input)

    # Build loaders for only PEMS03.npz using current dataloader utilities
    module = TimeSeriesDataModule(
        # dataset_name="PEMS03.npz", # matches filename basename discovered under ICML_datasets
        dataset_name="ETTh1.csv",
        data_dir=data_dir_final,
        batch_size=int(data_cfg.get("batch_size", 128)),
        val_batch_size=int(data_cfg.get("val_batch_size", data_cfg.get("batch_size", 128))),
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", device.type == "cuda")),
        normalize=bool(data_cfg.get("normalize", True)),
        filename=None,  # rely on dataset_name filter above
        train=False,
        val=False,
        test=True,
    )
    loaders = module.get_dataloaders()

    # Pick the requested split
    split_loader = None
    for group in loaders:
        # group.name is relative path like "PEMS/PEMS03.npz"; dataset_name filter already narrowed to one
        if args.split == "train":
            split_loader = group.train
        elif args.split == "val":
            split_loader = group.val
        else:
            split_loader = group.test
        break

    if split_loader is None:
        raise RuntimeError(f"Requested split '{args.split}' not available for PEMS03.npz")

    # Load encoder
    ckpt_path = resolve_path(cfg_path.parent, Path(args.encoder_checkpoint))
    if ckpt_path is None:
        ckpt_path = Path(args.encoder_checkpoint).expanduser().resolve()
    encoder = _load_encoder(config, ckpt_path, device=device)

    # Run forward pass for a few batches
    total = 0

    print(f"Running encoder on PEMS03 split='{args.split}' from {data_dir_final}")
    with torch.no_grad():
        for i, batch in enumerate(split_loader):
            seq_x = batch[0].to(device).float().transpose(1, 2)  # (B, T, F) -> (B, F, T)

            if seq_x.shape[1] > 1:
                outs = []
                for feat_idx in range(seq_x.shape[1]):
                    # pass each feature/channel separately through the encoder
                    emb = encoder(seq_x[:, feat_idx, :].unsqueeze(1))
                    outs.append(emb)
                # outs is list of tensors with shape (B, E, ...)
                stacked = torch.stack(outs, dim=-1)  # (B, F, E, ...)
                # flatten feature+embedding dims into (B, F * E * ...)
                out = stacked.swapaxes(1,2)
            else:
                # single feature channel: run directly
                out = encoder(seq_x)

            total += seq_x.size(0)
            print(f"batch {i}: x={tuple(seq_x.shape)} -> embedding={tuple(out.shape)}")
            if args.max_batches and args.max_batches > 0 and (i + 1) >= args.max_batches:
                break

    print(f"Done. Processed {total} samples.")

    # Plot encoder outputs against the input time series for one sample
    import matplotlib.pyplot as plt

    # Ensure we have at least one batch available (use last processed batch if present)
    if 'seq_x' not in globals() or 'out' not in globals():
        with torch.no_grad():
            for batch in split_loader:
                seq_x = batch[0].to(device).float().transpose(1, 2)
                if seq_x.shape[1] > 1:
                    outs = []
                    for feat_idx in range(seq_x.shape[1]):
                        emb = encoder(seq_x[:, feat_idx, :].unsqueeze(1))
                        outs.append(emb)
                    stacked = torch.stack(outs, dim=-1)
                    out = stacked.swapaxes(1, 2)
                else:
                    out = encoder(seq_x)
                break

    # Move tensors to CPU numpy arrays
    seq_np = seq_x.detach().cpu().numpy()  # shape (B, F, T)
    out_np = out.detach().cpu().numpy()

    B, F, T = seq_np.shape

    # Try to recover predictions in shape (B, F, T_pred)
    pred = None
    if out_np.ndim >= 3:
        # exact match (B, F, T)
        if out_np.shape[0] == B and out_np.shape[1] == F and out_np.shape[-1] == T:
            pred = out_np
        # match by collapsing middle dims into feature dim
        elif out_np.shape[0] == B and out_np.shape[-1] == T:
            mid = int(np.prod(out_np.shape[1:-1]))
            resh = out_np.reshape(B, mid, T)
            if mid == F:
                pred = resh
            elif mid % F == 0:
                pred = resh.reshape(B, F, -1)
        # if shape equals total elements, try reshape
        elif out_np.size == (B * F * T):
            pred = out_np.reshape(B, F, T)
    elif out_np.ndim == 2 and out_np.shape[0] == B and out_np.shape[1] == F * T:
        pred = out_np.reshape(B, F, T)

    # Plot the first sample in the batch
    sample_idx = 0
    plt.figure(figsize=(10, 4))
    x_in = np.arange(T)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Plot original input features
    for f in range(F):
        plt.plot(x_in, seq_np[sample_idx, f, :], color=colors[f % len(colors)], alpha=0.6, label=f"input_f{f}" if f < 8 else None)

    # Plot predictions if available
    if pred is not None:
        T_pred = pred.shape[2]
        # If prediction length equals input length, overlay directly
        if T_pred == T:
            for f in range(F):
                plt.plot(x_in, pred[sample_idx, f, :], linestyle="--", color=colors[f % len(colors)], linewidth=2, label=f"pred_f{f}" if f < 8 else None)
        else:
            # If prediction is a forecast extending beyond input, append on x-axis
            x_pred = np.arange(T, T + T_pred)
            for f in range(F):
                plt.plot(x_pred, pred[sample_idx, f, :], linestyle="--", color=colors[f % len(colors)], linewidth=2, label=f"pred_f{f}" if f < 8 else None)
    else:
        # Fallback: plot raw out values for the sample as points
        flat = out_np[sample_idx].ravel()
        xp = np.linspace(0, T - 1, len(flat))
        plt.plot(xp, flat, "xk", label="raw_prediction")

    plt.title("Input time series (solid) and encoder prediction (dashed)")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()