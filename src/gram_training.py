"""GRAM training entry-point.

Dual-encoder alignment with Gramian volume contrastive loss.
Structure mirrors CLIP-style training in cosine_training.py while replacing
objective with geometric volume-based logits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import training_utils as tu
import util as u
from cosine_training import _build_time_series_loaders
from moco_training import resolve_checkpoint_dir, resolve_path


def gram_volume_loss(
    z_ts: torch.Tensor,
    z_rp: torch.Tensor,
    temperature: float = 0.07,
    return_parts: bool = False,
):
    """Symmetric CE on negative Gramian volume logits.

    vol(i,j) = sqrt(1 - cos(i,j)^2)
    logits = -vol / temperature
    """
    # Do geometry in float32 for AMP stability.
    z_ts = z_ts.float()
    z_rp = z_rp.float()

    ts = F.normalize(z_ts, dim=-1, eps=1e-6)
    rp = F.normalize(z_rp, dim=-1, eps=1e-6)

    cos = torch.matmul(ts, rp.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    vol = torch.sqrt((1.0 - cos.pow(2)).clamp_min(1e-12))
    logits = -vol / max(float(temperature), 1e-6)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

    labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    if return_parts:
        return loss, logits, vol
    return loss


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRAM dual-encoder training")
    default_cfg = Path(__file__).resolve().parent / "configs" / "lotsa_gram.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--no_comet", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_component(path: Path, module: torch.nn.Module) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {path}")
    state = torch.load(path, map_location="cpu")
    module.load_state_dict(state["model_state_dict"])
    return state


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(config_path)

    experiment = None
    if not args.no_comet:
        from comet_utils import create_comet_experiment

        experiment = create_comet_experiment("gram")

    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    training_cfg = config.training
    epochs = args.epochs if args.epochs is not None else int(training_cfg.get("epochs", 200))
    noise_std = args.noise_std if args.noise_std is not None else float(training_cfg.get("noise_std", 0.01))
    lr = float(training_cfg.get("learning_rate", training_cfg.get("lr", 1e-3)))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    temperature = float(training_cfg.get("temperature", 0.2))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    use_amp = bool(training_cfg.get("use_amp", True))

    if experiment is not None:
        experiment.log_parameters(
            {
                "epochs": epochs,
                "noise_std": noise_std,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "temperature": temperature,
                "use_amp": use_amp,
                "max_grad_norm": max_grad_norm,
                "seed": config.seed,
            }
        )

    train_loader, val_loader, _ = _build_time_series_loaders(config_path, config.data, seed=config.seed)
    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    encoder = tu.build_encoder_from_config(config.model)
    visual_encoder = tu.build_visual_encoder_from_config(config.model)
    projection_dim = int(config.model.get("model_dim", 128))
    projection_head = u.build_projection_head(encoder, output_dim=projection_dim)
    visual_projection_head = u.build_projection_head(visual_encoder, output_dim=projection_dim)

    params = (
        list(encoder.parameters())
        + list(visual_encoder.parameters())
        + list(projection_head.parameters())
        + list(visual_projection_head.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    resume_dir = None
    initial_epoch = 0
    best_loss = None

    if args.resume_checkpoint is not None:
        resume_candidate = resolve_path(Path.cwd(), args.resume_checkpoint)
        if resume_candidate is None:
            raise FileNotFoundError(f"Unable to resolve resume path: {args.resume_checkpoint}")

        resume_dir = resume_candidate if resume_candidate.is_dir() else resume_candidate.parent
        ts_state = _load_component(resume_dir / "time_series_last.pt", encoder)
        _load_component(resume_dir / "visual_encoder_last.pt", visual_encoder)
        _load_component(resume_dir / "time_series_projection_last.pt", projection_head)
        _load_component(resume_dir / "visual_projection_last.pt", visual_projection_head)

        optimizer.load_state_dict(ts_state["optimizer_state_dict"])
        initial_epoch = min(epochs, int(ts_state.get("epoch", 0)) + 1)
        best_loss = float(ts_state.get("loss", float("inf")))
        print(f"Resuming from {resume_dir.resolve()} (epoch {initial_epoch}).")

    checkpoint_dir = (
        resume_dir.resolve() if resume_dir is not None else resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    )
    print(f"Checkpoints: {checkpoint_dir}")

    encoder.to(device)
    visual_encoder.to(device)
    projection_head.to(device)
    visual_projection_head.to(device)

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    best_metric = float("inf") if best_loss is None else float(best_loss)

    for epoch in range(initial_epoch, epochs):
        encoder.train()
        visual_encoder.train()
        projection_head.train()
        visual_projection_head.train()

        epoch_loss = 0.0
        batches = 0
        nan_vol_batches = 0
        skipped_nonfinite_batches = 0

        with torch.enable_grad():
            for batch in train_loader:
                if isinstance(batch, dict) and "target" in batch and "lengths" in batch:
                    padded = batch["target"].to(device).float()
                    lengths = batch["lengths"].to(device)
                    if (lengths == lengths[0]).all():
                        length = int(lengths[0].item())
                        x_q = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :length]))
                    else:
                        continue
                else:
                    seq = u.prepare_sequence(u.extract_sequence(batch)).to(device).float()
                    x_q = u.reshape_multivariate_series(seq)

                x_k = u.make_positive_view(x_q + noise_std * torch.randn_like(x_q))

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=amp_enabled):
                    q_proj = projection_head(encoder(x_q))
                    k_proj = visual_projection_head(visual_encoder(x_k))
                # Keep the loss in full precision to avoid half-precision edge cases.
                with autocast(enabled=False):
                    loss, logits, vol = gram_volume_loss(q_proj, k_proj, temperature, return_parts=True)

                if not torch.isfinite(loss):
                    skipped_nonfinite_batches += 1
                    continue
                if torch.isnan(vol).any():
                    nan_vol_batches += 1

                if amp_enabled:
                    scaler.scale(loss).backward()
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    optimizer.step()

                epoch_loss += float(loss.item())
                batches += 1

        if batches == 0:
            raise RuntimeError(
                f"No valid training batches in epoch={epoch + 1}; "
                f"skipped_nonfinite_batches={skipped_nonfinite_batches}"
            )

        train_loss = epoch_loss / max(1, batches)

        val_loss = None
        val_diag = None
        val_offdiag = None
        if val_loader is not None:
            encoder.eval()
            visual_encoder.eval()
            projection_head.eval()
            visual_projection_head.eval()

            val_total = 0.0
            val_batches = 0
            diag_vals = []
            off_vals = []
            with torch.no_grad():
                for val_batch in val_loader:
                    if isinstance(val_batch, dict) and "target" in val_batch and "lengths" in val_batch:
                        padded = val_batch["target"].to(device).float()
                        lengths = val_batch["lengths"].to(device)
                        if not (lengths == lengths[0]).all():
                            continue
                        length = int(lengths[0].item())
                        vx_q = u.reshape_multivariate_series(u.prepare_sequence(padded[:, :length]))
                    else:
                        vseq = u.prepare_sequence(u.extract_sequence(val_batch)).to(device).float()
                        vx_q = u.reshape_multivariate_series(vseq)

                    vx_k = u.make_positive_view(vx_q + noise_std * torch.randn_like(vx_q))
                    vq_proj = projection_head(encoder(vx_q))
                    vk_proj = visual_projection_head(visual_encoder(vx_k))
                    vloss, vlogits, _ = gram_volume_loss(vq_proj, vk_proj, temperature, return_parts=True)

                    if torch.isfinite(vloss):
                        val_total += float(vloss.item())
                        val_batches += 1
                        diag = torch.diag(vlogits)
                        off = vlogits[~torch.eye(vlogits.size(0), dtype=torch.bool, device=vlogits.device)]
                        diag_vals.append(float(diag.mean().item()))
                        off_vals.append(float(off.mean().item()))

            if val_batches > 0:
                val_loss = val_total / val_batches
                val_diag = sum(diag_vals) / len(diag_vals)
                val_offdiag = sum(off_vals) / len(off_vals)

        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch + 1)
            experiment.log_metric("nan_vol_batches", nan_vol_batches, step=epoch + 1)
            experiment.log_metric("skipped_nonfinite_batches", skipped_nonfinite_batches, step=epoch + 1)
            if val_loss is not None:
                experiment.log_metric("val_loss", val_loss, step=epoch + 1)
            if val_diag is not None and val_offdiag is not None:
                experiment.log_metric("val_diag_logit", val_diag, step=epoch + 1)
                experiment.log_metric("val_offdiag_logit", val_offdiag, step=epoch + 1)

        best_metric = u.log_and_save(
            optimizer,
            models={
                "time_series": encoder,
                "visual_encoder": visual_encoder,
                "time_series_projection": projection_head,
                "visual_projection": visual_projection_head,
            },
            epoch=epoch,
            epochs=epochs,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint_dir=checkpoint_dir,
            best_loss=best_metric,
        )

        if val_diag is not None and val_offdiag is not None:
            print(f"  logits sanity: diag={val_diag:.4f}, offdiag={val_offdiag:.4f}")
        if skipped_nonfinite_batches > 0:
            print(f"  skipped_nonfinite_batches={skipped_nonfinite_batches}")

    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
