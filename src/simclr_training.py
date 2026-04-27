"""SimCLR-style single-encoder contrastive training.

Trains *one* encoder (either temporal or visual) using two augmented views of
the same time series.  This is the single-modality counterpart of the dual-
encoder CLIP training in cosine_training.py.

  --mode temporal   →  MambaEncoder trained with SimCLR (two noisy temporal views)
  --mode visual     →  UpperTriDiagSimCLREncoder (anti-diagonal lag tokenization,
                       rp_mv_strategy=mean) trained with SimCLR (two noisy RP views)

The checkpoint layout mirrors cosine_training.py so that probe_lotsa_checkpoint.py
can load the encoder directly.

Usage
-----
    python3 src/simclr_training.py \
        --config src/configs/lotsa_simclr_temporal.yaml \
        --mode temporal

    python3 src/simclr_training.py \
        --config src/configs/lotsa_simclr_visual.yaml \
        --mode visual
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import comet_ml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import training_utils as tu
import util as u
from path_utils import resolve_path, resolve_checkpoint_dir


# ── helpers ─────────────────────────────────────────────────────────────────


def _resolve_data_root(config_path: Path, candidate: Optional[object]) -> Path:
    resolved = resolve_path(config_path.parent, candidate)
    if resolved is not None:
        return resolved
    return (config_path.parent.parent / "data").resolve()


def _build_loaders(
    config_path: Path,
    data_cfg: Dict[str, object],
    *,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Thin wrapper — reuses the same loader logic as cosine_training.py."""
    import yaml
    from typing import Sequence

    dataset_type = str(data_cfg.get("dataset_type", "cronos")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))
    normalize = bool(data_cfg.get("normalize", True))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio_cfg = data_cfg.get("val_ratio")
    val_ratio = float(val_ratio_cfg) if val_ratio_cfg is not None else 0.2

    cronos_kwargs: Dict[str, object] = dict(data_cfg.get("cronos_kwargs", {}) or {})
    datasets_spec = data_cfg.get("datasets")
    dataset_name = data_cfg.get("dataset_name")
    data_root = _resolve_data_root(config_path, data_cfg.get("data_dir"))

    if dataset_type == "cronos":
        load_kwargs = cronos_kwargs.setdefault("load_kwargs", {})
        load_kwargs.setdefault("offline_cache_dir", str(data_root))
        load_kwargs.setdefault("force_offline", True)

    train_loader, val_loader = u.build_time_series_dataloaders(
        data_dir=str(data_root),
        filename=data_cfg.get("filename"),
        dataset_name=dataset_name,
        datasets=datasets_spec,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        dataset_type=dataset_type,
        val_split=data_cfg.get("val_split"),
        seed=seed,
        cronos_kwargs=cronos_kwargs,
    )
    return train_loader, val_loader


def _simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric NT-Xent loss (SimCLR).

    Both z1 and z2 are L2-normalised embeddings of shape (N, D).
    Positive pairs: (i, i) across the two views.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
    loss_12 = F.cross_entropy(logits, targets)
    loss_21 = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_12 + loss_21)


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head (SimCLR-style)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── training loop ────────────────────────────────────────────────────────────


def run_simclr_training(
    *,
    encoder: nn.Module,
    projection_head: nn.Module,
    mode: str,                      # "temporal" or "visual"
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int = 100,
    noise_std: float = 0.01,
    optimizer: torch.optim.Optimizer,
    temperature: float = 0.07,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    experiment: Optional[object] = None,
) -> None:
    encoder.to(device)
    projection_head.to(device)

    for state in optimizer.state.values():
        for key, val in list(state.items()):
            if torch.is_tensor(val):
                state[key] = val.to(device)

    best_metric = float("inf") if best_loss is None else float(best_loss)
    start_epoch = max(0, int(initial_epoch))
    if start_epoch >= epochs:
        print(f"Already completed {start_epoch}/{epochs} epochs — nothing to do.")
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _encode(x: torch.Tensor) -> torch.Tensor:
        """Run the (single) encoder on a batch."""
        if mode == "temporal":
            return encoder(x)
        else:
            # visual encoder expects (B, F, T)
            return encoder(x)

    def _augment(x: torch.Tensor) -> torch.Tensor:
        """Return a second augmented view by adding Gaussian noise."""
        return x + noise_std * torch.randn_like(x)

    for epoch in range(start_epoch, epochs):
        encoder.train()
        projection_head.train()

        epoch_loss = 0.0
        batches = 0
        total = len(train_loader) if hasattr(train_loader, "__len__") else None
        desc = f"Epoch {epoch + 1}/{epochs}"

        with tqdm(train_loader, desc=desc, total=total) as pbar:
            for batch in pbar:
                if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
                    padded = batch["target"].to(device).float()
                    lengths = batch["lengths"].to(device)
                    has_two_views = "target2" in batch
                    padded2 = batch["target2"].to(device).float() if has_two_views else None

                    if (lengths == lengths[0]).all():
                        L = int(lengths[0].item())
                        x1 = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
                        if has_two_views:
                            x2 = u.reshape_multivariate_series(u.prepare_sequence(padded2[:, :L]))
                        else:
                            x2 = _augment(x1)
                        z1 = projection_head(_encode(x1))
                        z2 = projection_head(_encode(x2))
                        loss = _simclr_loss(z1, z2, temperature)
                    else:
                        z1_list, z2_list = [], []
                        for i in range(padded.size(0)):
                            Li = int(lengths[i].item())
                            if Li < 2:
                                continue
                            xi1 = u.reshape_multivariate_series(
                                u.prepare_sequence(padded[i, :Li].unsqueeze(0))
                            )
                            if has_two_views:
                                xi2 = u.reshape_multivariate_series(
                                    u.prepare_sequence(padded2[i, :Li].unsqueeze(0))
                                )
                            else:
                                xi2 = _augment(xi1)
                            z1_list.append(projection_head(_encode(xi1)))
                            z2_list.append(projection_head(_encode(xi2)))
                        if not z1_list:
                            continue
                        z1 = torch.cat(z1_list, dim=0)
                        z2 = torch.cat(z2_list, dim=0)
                        loss = _simclr_loss(z1, z2, temperature)
                else:
                    seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
                    x1 = u.reshape_multivariate_series(seq)
                    x2 = _augment(x1)
                    z1 = projection_head(_encode(x1))
                    z2 = projection_head(_encode(x2))
                    loss = _simclr_loss(z1, z2, temperature)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                batches += 1
                pbar.set_postfix(
                    batch_loss=f"{batch_loss:.4f}",
                    avg=f"{(epoch_loss / batches):.4f}",
                )

        train_loss = epoch_loss / batches if batches > 0 else float("nan")

        val_loss = None
        if val_loader is not None:
            encoder.eval()
            projection_head.eval()
            val_total, val_batches = 0.0, 0
            with torch.no_grad():
                for val_batch in val_loader:
                    if isinstance(val_batch, dict) and "target" in val_batch and "lengths" in val_batch:
                        padded = val_batch["target"].to(device).float()
                        lengths = val_batch["lengths"].to(device)
                        if (lengths == lengths[0]).all():
                            L = int(lengths[0].item())
                            vx1 = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :L]))
                            vx2 = _augment(vx1)
                            vz1 = projection_head(_encode(vx1))
                            vz2 = projection_head(_encode(vx2))
                            val_total += _simclr_loss(vz1, vz2, temperature).item()
                            val_batches += 1
                    else:
                        vseq = u.prepare_sequence(u.extract_sequence(val_batch)).to(device).float()
                        vx1 = u.reshape_multivariate_series(vseq)
                        vx2 = _augment(vx1)
                        vz1 = projection_head(_encode(vx1))
                        vz2 = projection_head(_encode(vx2))
                        val_total += _simclr_loss(vz1, vz2, temperature).item()
                        val_batches += 1
            val_loss = val_total / val_batches if val_batches > 0 else None
            encoder.train()
            projection_head.train()

        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            experiment.log_metric("learning_rate", optimizer.param_groups[0]["lr"], step=epoch + 1)

        # Save checkpoints using the same naming convention as cosine_training.py
        # so probe_lotsa_checkpoint.py can load them.
        ckpt_name = "time_series" if mode == "temporal" else "visual_encoder"
        monitor_loss = val_loss if val_loss is not None else train_loss
        is_best = monitor_loss < best_metric
        if is_best:
            best_metric = monitor_loss

        def _save(suffix: str) -> None:
            path = checkpoint_dir / f"{ckpt_name}_{suffix}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": monitor_loss,
                },
                path,
            )
            proj_path = checkpoint_dir / f"{ckpt_name}_projection_{suffix}.pt"
            torch.save({"epoch": epoch, "model_state_dict": projection_head.state_dict()}, proj_path)

        _save("last")
        if is_best:
            _save("best")

        best_tag = " [best]" if is_best else ""
        val_str = f"  val={val_loss:.4f}" if val_loss is not None else ""
        print(f"Epoch {epoch + 1}/{epochs}  train={train_loss:.4f}{val_str}{best_tag}")

    print(f"SimCLR training complete. Best loss: {best_metric:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR single-encoder contrastive training")
    default_cfg = Path(__file__).resolve().parent / "configs" / "lotsa_simclr_temporal.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument(
        "--mode",
        choices=["temporal", "visual"],
        default=None,
        help="Which encoder to train. Overrides config if provided.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = tu.load_config(config_path)

    # mode: CLI flag > config field > error
    mode_from_config = str(config.model.get("rp_encoder", config.model.get("simclr_mode", ""))).lower()
    # Normalize: "upper_tri" → "visual" for mode routing
    if mode_from_config == "upper_tri":
        mode_from_config = "visual"
    if args.mode is not None:
        mode = args.mode
    elif mode_from_config in ("temporal", "visual"):
        mode = mode_from_config
    else:
        raise ValueError(
            "Encoder mode not specified. Pass --mode temporal or --mode visual, "
            "or set model.rp_encoder in the config YAML."
        )

    from comet_utils import create_comet_experiment
    exp_name = str(getattr(config, "experiment_name", f"simclr_{mode}"))
    comet_key_map = {
        "ts_simclr_temporal_lotsa": "simclr_temporal",
        "ts_simclr_visual_lotsa": "simclr_visual",
    }
    comet_key = comet_key_map.get(exp_name, f"simclr_{mode}")
    experiment = create_comet_experiment(comet_key)

    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Mode: {mode}  |  Device: {device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 100))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.01))
    temperature = float(training_cfg.get("temperature", 0.07))
    lr = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    experiment.log_parameters({
        "mode": mode,
        "epochs": epochs,
        "noise_std": noise_std,
        "temperature": temperature,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": config.seed,
    })

    train_loader, val_loader = _build_loaders(config_path, config.data, seed=config.seed)
    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    model_cfg = config.model
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    model_dim = int(model_cfg.get("model_dim", 256))

    if mode == "temporal":
        encoder = tu.build_encoder_from_config(model_cfg)
        encoder_out_dim = embedding_dim
        print(f"Temporal encoder: MambaEncoder  (embedding_dim={encoder_out_dim})")
    else:
        from models.mamba_visual_encoder import UpperTriDiagRPEncoder
        encoder = UpperTriDiagRPEncoder(
            patch_len=int(model_cfg.get("input_dim", 32)),
            d_model=model_dim,
            n_layers=int(model_cfg.get("depth", 8)),
            embedding_dim=embedding_dim,
            rp_mv_strategy="mean",       # best from Ablation A
        )
        encoder_out_dim = embedding_dim
        print(f"Visual encoder: UpperTriDiagRPEncoder (patch_len={model_cfg.get('input_dim', 32)}, d_model={model_dim}, n_layers={model_cfg.get('depth', 8)}, embedding_dim={encoder_out_dim})")

    projection_head = ProjectionHead(
        in_dim=encoder_out_dim,
        hidden_dim=model_dim,
        out_dim=model_dim,
    )

    params = list(encoder.parameters()) + list(projection_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    initial_epoch = 0
    best_loss: Optional[float] = None
    resume_dir: Optional[Path] = None

    if args.resume_checkpoint is not None:
        resume_candidate = resolve_path(Path.cwd(), args.resume_checkpoint)
        if resume_candidate is None:
            raise FileNotFoundError(f"Cannot resolve resume path: {args.resume_checkpoint}")
        resume_dir = resume_candidate if resume_candidate.is_dir() else resume_candidate.parent
        ckpt_name = "time_series" if mode == "temporal" else "visual_encoder"
        last_path = resume_dir / f"{ckpt_name}_last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {last_path}")
        state = torch.load(last_path, map_location="cpu")
        encoder.load_state_dict(state["model_state_dict"])
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            print(f"Resumed model AND optimizer from {last_path}")
        except Exception as e:
            print(f"Warning: Could not resume optimizer state ({e}). Initializing fresh optimizer.")
        initial_epoch = min(epochs, int(state.get("epoch", 0)) + 1)
        best_loss = state.get("loss")
        print(f"Resumed from {last_path} (epoch {initial_epoch})")

    checkpoint_dir = (
        resume_dir.resolve()
        if resume_dir is not None
        else resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    )
    print(f"Checkpoints: {checkpoint_dir}")

    run_simclr_training(
        encoder=encoder,
        projection_head=projection_head,
        mode=mode,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        optimizer=optimizer,
        temperature=temperature,
        initial_epoch=initial_epoch,
        best_loss=best_loss,
        experiment=experiment,
    )

    experiment.end()


if __name__ == "__main__":
    main()
