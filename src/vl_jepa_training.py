"""VL-JEPA style cross-modal predictive training.

Joint Embedding Predictive Architecture adapted for time series + recurrence plot
alignment. Unlike CLIP (contrastive) or BYOL (same-modal two-views), VL-JEPA
trains each modality's online encoder to predict the other modality's EMA target
representation with no negative pairs.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import training_utils as tu
import util as u
from byol_training import MLPHead, _build_loaders, _update_ema
from moco_training import resolve_checkpoint_dir, resolve_path
from models.mamba_visual_encoder import UpperTriDiagRPEncoder


def vl_jepa_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """L2 predictive loss in the full latent space."""
    return F.mse_loss(p, z)


def _prepare_batch(batch, device: torch.device) -> Optional[torch.Tensor]:
    """Return tensor shaped (B, F, T) or None for unsupported variable-length batch."""
    if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
        padded = batch["target"].to(device).float()
        lengths = batch["lengths"].to(device)
        if not (lengths == lengths[0]).all():
            return None
        length = int(lengths[0].item())
        seq = u.prepare_sequence(padded[:, :length])
    else:
        seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
    return u.reshape_multivariate_series(seq)


def _save_component(
    checkpoint_dir: Path,
    name: str,
    epoch: int,
    monitor_loss: float,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    ema_model: Optional[nn.Module],
    suffix: str,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": monitor_loss,
    }
    if ema_model is not None:
        state["ema_state_dict"] = ema_model.state_dict()
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, checkpoint_dir / f"{name}_{suffix}.pt")


def run_vl_jepa_training(
    *,
    ts_encoder: nn.Module,
    rp_encoder: nn.Module,
    ts_projector: nn.Module,
    rp_projector: nn.Module,
    ts_predictor: nn.Module,
    rp_predictor: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    checkpoint_dir: Path,
    epochs: int,
    noise_std: float,
    max_grad_norm: float,
    optimizer: torch.optim.Optimizer,
    ema_tau_base: float,
    initial_epoch: int = 0,
    best_loss: Optional[float] = None,
    experiment: Optional[object] = None,
    ts_encoder_ema_state: Optional[dict] = None,
    rp_encoder_ema_state: Optional[dict] = None,
    ts_projector_ema_state: Optional[dict] = None,
    rp_projector_ema_state: Optional[dict] = None,
) -> None:
    """Main VL-JEPA training loop."""

    ts_encoder.to(device)
    rp_encoder.to(device)
    ts_projector.to(device)
    rp_projector.to(device)
    ts_predictor.to(device)
    rp_predictor.to(device)

    ts_encoder_ema = copy.deepcopy(ts_encoder).to(device)
    rp_encoder_ema = copy.deepcopy(rp_encoder).to(device)
    ts_projector_ema = copy.deepcopy(ts_projector).to(device)
    rp_projector_ema = copy.deepcopy(rp_projector).to(device)

    if ts_encoder_ema_state is not None:
        ts_encoder_ema.load_state_dict(ts_encoder_ema_state)
    if rp_encoder_ema_state is not None:
        rp_encoder_ema.load_state_dict(rp_encoder_ema_state)
    if ts_projector_ema_state is not None:
        ts_projector_ema.load_state_dict(ts_projector_ema_state)
    if rp_projector_ema_state is not None:
        rp_projector_ema.load_state_dict(rp_projector_ema_state)

    for param in (
        list(ts_encoder_ema.parameters())
        + list(rp_encoder_ema.parameters())
        + list(ts_projector_ema.parameters())
        + list(rp_projector_ema.parameters())
    ):
        param.requires_grad_(False)

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf") if best_loss is None else float(best_loss)
    start_epoch = max(0, int(initial_epoch))
    if start_epoch >= epochs:
        print(f"Requested epochs already completed ({start_epoch}/{epochs}).")
        return

    total_steps = max(1, (epochs - start_epoch) * max(1, len(train_loader)))
    global_step = start_epoch * max(1, len(train_loader))

    def _augment(x: torch.Tensor) -> torch.Tensor:
        return x + noise_std * torch.randn_like(x)

    for epoch in range(start_epoch, epochs):
        ts_encoder.train()
        rp_encoder.train()
        ts_projector.train()
        rp_projector.train()
        ts_predictor.train()
        rp_predictor.train()

        epoch_loss = 0.0
        batches = 0
        collapse_count = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for batch in pbar:
                tau = 1.0 - (1.0 - ema_tau_base) * (
                    (1.0 + torch.cos(torch.tensor(torch.pi * global_step / total_steps)).item()) / 2.0
                )

                x = _prepare_batch(batch, device)
                if x is None:
                    global_step += 1
                    continue
                x = _augment(x)

                z_ts_online = ts_projector(ts_encoder(x))
                z_rp_online = rp_projector(rp_encoder(x))
                p_ts = ts_predictor(z_ts_online)
                p_rp = rp_predictor(z_rp_online)

                with torch.no_grad():
                    z_rp_target = rp_projector_ema(rp_encoder_ema(x))
                    z_ts_target = ts_projector_ema(ts_encoder_ema(x))

                loss = 0.5 * (vl_jepa_loss(p_ts, z_rp_target) + vl_jepa_loss(p_rp, z_ts_target))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite VL-JEPA loss at epoch={epoch + 1}, step={global_step}")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(ts_encoder.parameters())
                        + list(rp_encoder.parameters())
                        + list(ts_projector.parameters())
                        + list(rp_projector.parameters())
                        + list(ts_predictor.parameters())
                        + list(rp_predictor.parameters()),
                        max_grad_norm,
                    )

                optimizer.step()

                _update_ema(ts_encoder, ts_encoder_ema, tau)
                _update_ema(ts_projector, ts_projector_ema, tau)
                _update_ema(rp_encoder, rp_encoder_ema, tau)
                _update_ema(rp_projector, rp_projector_ema, tau)

                loss_value = float(loss.item())
                epoch_loss += loss_value
                batches += 1
                global_step += 1

                with torch.no_grad():
                    z_norm = z_rp_target.norm(dim=-1).mean().item()
                    if z_norm <= 0.01:
                        collapse_count += 1

                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{epoch_loss / max(1, batches):.4f}",
                    tau=f"{tau:.4f}",
                    z_norm=f"{z_norm:.3f}",
                )

        train_loss = epoch_loss / max(1, batches)

        val_loss = None
        if val_loader is not None:
            ts_encoder.eval()
            rp_encoder.eval()
            ts_projector.eval()
            rp_projector.eval()
            ts_predictor.eval()
            rp_predictor.eval()

            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = _prepare_batch(batch, device)
                    if x is None:
                        continue
                    x = _augment(x)

                    z_rp_target = rp_projector_ema(rp_encoder_ema(x))
                    z_ts_target = ts_projector_ema(ts_encoder_ema(x))
                    p_ts = ts_predictor(ts_projector(ts_encoder(x)))
                    p_rp = rp_predictor(rp_projector(rp_encoder(x)))
                    val_batch_loss = 0.5 * (
                        vl_jepa_loss(p_ts, z_rp_target) + vl_jepa_loss(p_rp, z_ts_target)
                    )
                    if torch.isfinite(val_batch_loss):
                        val_total += float(val_batch_loss.item())
                        val_batches += 1

            if val_batches > 0:
                val_loss = val_total / val_batches

            ts_encoder.train()
            rp_encoder.train()
            ts_projector.train()
            rp_projector.train()
            ts_predictor.train()
            rp_predictor.train()

        monitor_loss = val_loss if val_loss is not None else train_loss
        is_best = monitor_loss < best_metric
        if is_best:
            best_metric = monitor_loss

        # Probe-compatible files
        _save_component(
            checkpoint_dir,
            "time_series",
            epoch,
            monitor_loss,
            ts_encoder,
            optimizer,
            ts_encoder_ema,
            "last",
        )
        _save_component(
            checkpoint_dir,
            "visual_encoder",
            epoch,
            monitor_loss,
            rp_encoder,
            None,
            rp_encoder_ema,
            "last",
        )
        _save_component(
            checkpoint_dir,
            "time_series_projection",
            epoch,
            monitor_loss,
            ts_projector,
            None,
            ts_projector_ema,
            "last",
        )
        _save_component(
            checkpoint_dir,
            "visual_projection",
            epoch,
            monitor_loss,
            rp_projector,
            None,
            rp_projector_ema,
            "last",
        )

        # Extra files for exact resume
        _save_component(
            checkpoint_dir,
            "time_series_predictor",
            epoch,
            monitor_loss,
            ts_predictor,
            None,
            None,
            "last",
        )
        _save_component(
            checkpoint_dir,
            "visual_predictor",
            epoch,
            monitor_loss,
            rp_predictor,
            None,
            None,
            "last",
        )

        if is_best:
            for name, model, ema_model in [
                ("time_series", ts_encoder, ts_encoder_ema),
                ("visual_encoder", rp_encoder, rp_encoder_ema),
                ("time_series_projection", ts_projector, ts_projector_ema),
                ("visual_projection", rp_projector, rp_projector_ema),
            ]:
                _save_component(
                    checkpoint_dir,
                    name,
                    epoch,
                    monitor_loss,
                    model,
                    optimizer if name == "time_series" else None,
                    ema_model,
                    "best",
                )
            _save_component(
                checkpoint_dir,
                "time_series_predictor",
                epoch,
                monitor_loss,
                ts_predictor,
                None,
                None,
                "best",
            )
            _save_component(
                checkpoint_dir,
                "visual_predictor",
                epoch,
                monitor_loss,
                rp_predictor,
                None,
                None,
                "best",
            )

        best_tag = " [best]" if is_best else ""
        val_txt = f"  val={val_loss:.4f}" if val_loss is not None else ""
        print(
            f"Epoch {epoch + 1}/{epochs}  train={train_loss:.4f}{val_txt}{best_tag}"
            f"  collapse_warn_steps={collapse_count}"
        )

        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            experiment.log_metric("ema_tau", tau, step=epoch + 1)
            experiment.log_metric("collapse_warn_steps", collapse_count, step=epoch + 1)

    print(f"VL-JEPA training complete. Best loss: {best_metric:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VL-JEPA cross-modal predictive training")
    default_cfg = Path(__file__).resolve().parent / "configs" / "lotsa_vl_jepa.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--no_comet", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_state(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return torch.load(path, map_location="cpu")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = tu.load_config(config_path)

    experiment = None
    if not args.no_comet:
        from comet_utils import create_comet_experiment

        experiment = create_comet_experiment("vl_jepa")

    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Starting VL-JEPA training - device={device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 200))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.05))
    lr = float(training_cfg.get("learning_rate", training_cfg.get("lr", 3e-4)))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    ema_tau = float(training_cfg.get("ema_tau", 0.996))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))

    if experiment is not None:
        experiment.log_parameters(
            {
                "epochs": epochs,
                "noise_std": noise_std,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "ema_tau": ema_tau,
                "seed": config.seed,
                "max_grad_norm": max_grad_norm,
            }
        )

    train_loader, val_loader = _build_loaders(config_path, config.data, seed=config.seed)
    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    model_cfg = config.model
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    model_dim = int(model_cfg.get("model_dim", 256))
    proj_dim = int(model_cfg.get("proj_dim", 256))
    pred_dim = int(model_cfg.get("pred_dim", 128))

    ts_encoder = tu.build_encoder_from_config(model_cfg)
    rp_encoder = UpperTriDiagRPEncoder(
        patch_len=int(model_cfg.get("input_dim", 64)),
        d_model=model_dim,
        n_layers=int(model_cfg.get("depth", 8)),
        embedding_dim=embedding_dim,
        rp_mv_strategy=str(model_cfg.get("rp_mv_strategy", "mean")),
    )
    print(f"TS encoder: MambaEncoder (embedding_dim={embedding_dim})")
    print(f"RP encoder: UpperTriDiagRPEncoder (embedding_dim={embedding_dim})")

    ts_projector = MLPHead(embedding_dim, proj_dim, proj_dim)
    rp_projector = MLPHead(embedding_dim, proj_dim, proj_dim)
    ts_predictor = MLPHead(proj_dim, pred_dim, proj_dim)
    rp_predictor = MLPHead(proj_dim, pred_dim, proj_dim)

    params = (
        list(ts_encoder.parameters())
        + list(rp_encoder.parameters())
        + list(ts_projector.parameters())
        + list(rp_projector.parameters())
        + list(ts_predictor.parameters())
        + list(rp_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    initial_epoch = 0
    best_loss_val: Optional[float] = None
    resume_dir: Optional[Path] = None

    ts_encoder_ema_state = None
    rp_encoder_ema_state = None
    ts_projector_ema_state = None
    rp_projector_ema_state = None

    if args.resume_checkpoint is not None:
        resume_candidate = resolve_path(Path.cwd(), args.resume_checkpoint)
        if resume_candidate is None:
            raise FileNotFoundError(f"Cannot resolve resume path: {args.resume_checkpoint}")
        resume_dir = resume_candidate if resume_candidate.is_dir() else resume_candidate.parent

        ts_state = _load_state(resume_dir / "time_series_last.pt")
        rp_state = _load_state(resume_dir / "visual_encoder_last.pt")
        ts_proj_state = _load_state(resume_dir / "time_series_projection_last.pt")
        rp_proj_state = _load_state(resume_dir / "visual_projection_last.pt")
        ts_pred_state = _load_state(resume_dir / "time_series_predictor_last.pt")
        rp_pred_state = _load_state(resume_dir / "visual_predictor_last.pt")

        ts_encoder.load_state_dict(ts_state["model_state_dict"])
        rp_encoder.load_state_dict(rp_state["model_state_dict"])
        ts_projector.load_state_dict(ts_proj_state["model_state_dict"])
        rp_projector.load_state_dict(rp_proj_state["model_state_dict"])
        ts_predictor.load_state_dict(ts_pred_state["model_state_dict"])
        rp_predictor.load_state_dict(rp_pred_state["model_state_dict"])

        optimizer.load_state_dict(ts_state["optimizer_state_dict"])

        ts_encoder_ema_state = ts_state.get("ema_state_dict")
        rp_encoder_ema_state = rp_state.get("ema_state_dict")
        ts_projector_ema_state = ts_proj_state.get("ema_state_dict")
        rp_projector_ema_state = rp_proj_state.get("ema_state_dict")

        initial_epoch = min(epochs, int(ts_state.get("epoch", 0)) + 1)
        best_loss_val = float(ts_state.get("loss", float("inf")))
        print(f"Resumed from {resume_dir} (epoch {initial_epoch})")

    checkpoint_dir = (
        resume_dir.resolve() if resume_dir is not None else resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    )
    print(f"Config: {config_path.name}")
    print(f"Checkpoints: {checkpoint_dir}")

    run_vl_jepa_training(
        ts_encoder=ts_encoder,
        rp_encoder=rp_encoder,
        ts_projector=ts_projector,
        rp_projector=rp_projector,
        ts_predictor=ts_predictor,
        rp_predictor=rp_predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        noise_std=noise_std,
        max_grad_norm=max_grad_norm,
        optimizer=optimizer,
        ema_tau_base=ema_tau,
        initial_epoch=initial_epoch,
        best_loss=best_loss_val,
        experiment=experiment,
        ts_encoder_ema_state=ts_encoder_ema_state,
        rp_encoder_ema_state=rp_encoder_ema_state,
        ts_projector_ema_state=ts_projector_ema_state,
        rp_projector_ema_state=rp_projector_ema_state,
    )

    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
