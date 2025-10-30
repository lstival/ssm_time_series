"""Lightweight cosine-similarity training loop using Chronos datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

import training_utils as tu
import util as u
from tqdm.auto import tqdm
from moco_training import (
    MoCoProjectionHead,
    build_dataloaders,
    infer_feature_dim,
    prepare_dataset,
    resolve_path,
    split_dataset,
)

def make_positive_view(
    x: torch.Tensor,
    *,
    noise_std: float = 0.1,
    jitter_std: float = 0.1,
    mask_prob: float = 0.20,
    scale_range: tuple[float, float] = (0.9, 1.1),
    time_dropout_prob: float = 0.1,
    max_dropout_ratio: float = 0.2,
    permute_segments: int = 1,
) -> torch.Tensor:
    """Create an augmented positive view from input time-series tensor.

    Expected input shape: (batch, channels, length).
    Augmentations applied (in order):
      - additive gaussian noise
      - small jitter (additional gaussian)
      - per-(sample,channel) scaling
      - random element masking
      - optional random temporal dropout (contiguous segment zeroed)
      - optional permutation of contiguous segments

    Returns a new tensor (does not modify input in-place).
    """
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.dim() != 3:
        raise ValueError("x must have shape (batch, channels, length)")

    out = x.clone()

    # additive noise + jitter
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    if jitter_std > 0:
        out = out + jitter_std * torch.randn_like(out)

    # per-sample, per-channel scaling
    b, c, L = out.shape
    device = out.device
    scale_min, scale_max = scale_range
    if not (scale_min == 1.0 and scale_max == 1.0):
        scales = torch.empty((b, c, 1), device=device).uniform_(scale_min, scale_max)
        out = out * scales

    # random element masking
    if mask_prob > 0:
        mask = torch.rand_like(out) < float(mask_prob)
        out = out.masked_fill(mask, 0.0)

    # time dropout: zero out a contiguous segment per sample with some probability
    if time_dropout_prob > 0 and L > 0:
        for i in range(b):
            if torch.rand(1, device=device).item() < float(time_dropout_prob):
                max_len = max(1, int(L * float(max_dropout_ratio)))
                drop_len = torch.randint(1, max_len + 1, (1,), device=device).item()
                start = torch.randint(0, L - drop_len + 1, (1,), device=device).item()
                out[i, :, start : start + drop_len] = 0.0

    # permutation of segments: split into K segments and shuffle their order
    if permute_segments and L > 1:
        K = int(permute_segments)
        K = max(2, min(K, L))  # at least 2 segments, at most L
        # compute segment boundaries (as even as possible)
        sizes = [L // K] * K
        for idx in range(L % K):
            sizes[idx] += 1
        boundaries = []
        pos = 0
        for s in sizes:
            boundaries.append((pos, pos + s))
            pos += s
        perm = torch.randperm(K, device=device).tolist()
        permuted = torch.empty_like(out)
        for si, pj in enumerate(perm):
            start_src, end_src = boundaries[pj]
            start_dst, end_dst = boundaries[si]
            permuted[:, :, start_dst:end_dst] = out[:, :, start_src:end_src]
        out = permuted

    return out


def clip_contrastive_loss(
    xq: torch.Tensor, xk: torch.Tensor, *, temperature: float = 0.07
) -> torch.Tensor:
    """CLIP-style symmetric cross-entropy loss.

    xq, xk: (batch, dim) feature tensors. The i-th row in xq should match the i-th row in xk.
    Returns scalar loss = 0.5 * (CE(logits_qk, targets) + CE(logits_kq, targets))
    where logits_qk = xq @ xk.T / temperature.
    """
    if not torch.is_tensor(xq) or not torch.is_tensor(xk):
        raise TypeError("xq and xk must be torch.Tensors")
    if xq.dim() != 2 or xk.dim() != 2:
        raise ValueError("xq and xk must be 2D tensors of shape (batch, dim)")
    if xq.shape[0] != xk.shape[0]:
        raise ValueError("xq and xk must have the same batch size")

    # normalize to unit length (important for cosine-similarity logits)
    q = nn.functional.normalize(xq, dim=1)
    k = nn.functional.normalize(xk, dim=1)

    # logits: (batch, batch)
    logits = torch.matmul(q, k.transpose(0, 1)) / float(temperature)

    targets = torch.arange(logits.size(0), device=logits.device, dtype=torch.long).to(xq.device)

    loss_q = nn.functional.cross_entropy(logits, targets)
    loss_k = nn.functional.cross_entropy(logits.transpose(0, 1), targets)

    return 0.5 * (loss_q + loss_k)


def run_cosine_training(
    *,
    encoder: nn.Module,
    visual_encoder: nn.Module,
    projection_head: nn.Module,
    visual_projection_head: nn.Module,
    train_loader,
    device: torch.device,
    epochs: int = 2,
    noise_std: float = 0.01,
) -> None:
    """Training loop that maximizes cosine similarity between two noisy views.

    Optimizer: AdamW with lr=1e-3.
    Loss: 1 - mean(cosine_similarity(q_proj, k_proj))
    """

    encoder.to(device).train()
    visual_encoder.to(device).train()
    projection_head.to(device).train()
    visual_projection_head.to(device).train()

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(visual_encoder.parameters()) +
        list(projection_head.parameters()) + list(visual_projection_head.parameters()),
        lr=1e-3
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0

        total = len(train_loader) if hasattr(train_loader, "__len__") else None
        desc = f"Epoch {epoch + 1}/{epochs}"
        with tqdm(train_loader, desc=desc, total=total) as pbar:
            for batch in pbar:
                seq = u.prepare_sequence(u.extract_sequence(batch)).to(device)

                x_q = seq.swapaxes(1, 2)
                noise = noise_std * torch.randn_like(x_q)
                x_k = x_q + noise
                x_k = make_positive_view(x_k)

                optimizer.zero_grad()

                q_encoded = encoder(x_q)
                k_encoded = visual_encoder(x_k)

                q_proj = projection_head(q_encoded)
                k_proj = visual_projection_head(k_encoded)

                q_proj = nn.functional.normalize(q_proj, dim=1)
                k_proj = nn.functional.normalize(k_proj, dim=1)

                # cosine_sim = nn.functional.cosine_similarity(q_proj, k_proj, dim=1)
                # loss = (1.0 - cosine_sim).mean()
                loss = clip_contrastive_loss(q_proj, k_proj)

                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                batches += 1

                pbar.set_postfix(batch_loss=f"{batch_loss:.4f}", avg_loss=f"{(epoch_loss / batches):.4f}")

        avg_loss = epoch_loss / batches if batches > 0 else float("nan")
        print(f"Epoch {epoch + 1}/{epochs} - Average loss: {avg_loss:.4f}")


def build_projection_head(encoder: nn.Module) -> nn.Module:
    """Infer encoder output dimension and create a projection head."""
    try:
        output_dim = encoder.final_norm._parameters["weight"].shape[0]
    except AttributeError as exc:
        raise RuntimeError("Encoder is expected to expose final_norm with learnable weight") from exc

    return MoCoProjectionHead(output_dim, output_dim, 128)


def main(argv: Optional[Iterable[str]] = None) -> None:
    default_cfg = Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml"
    config_path = resolve_path(Path.cwd(), default_cfg)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {default_cfg}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    data_cfg = config.data
    dataset = prepare_dataset(config_path, data_cfg)

    val_ratio = float(data_cfg.get("val_ratio", 0.0))
    train_dataset, _ = split_dataset(dataset, val_ratio=val_ratio, seed=config.seed)

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader, _ = build_dataloaders(
        train_dataset,
        None,
        batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    feature_dim = infer_feature_dim(train_loader)
    print(f"Inferred feature dimension: {feature_dim}")

    encoder = tu.build_encoder_from_config(config.model)
    visual_encoder = tu.build_visual_encoder_from_config(config.model)

    projection_head = build_projection_head(encoder)
    visual_projection_head = build_projection_head(visual_encoder)

    run_cosine_training(
        encoder=encoder,
        visual_encoder=visual_encoder,
        projection_head=projection_head,
        visual_projection_head=visual_projection_head,
        train_loader=train_loader,
        device=device,
        epochs=2,
    )


if __name__ == "__main__":
    main()

