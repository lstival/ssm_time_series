# filepath: c:\WUR\ssm_time_series\src\CLIP_training.py
"""Simplified CLIP-style training script (time-series + visual) based on cosine_training.py.

This keeps the same core training method:
- build dataloaders via util.build_time_series_dataloaders
- build encoders/projection heads via training_utils + util
- use a learnable CLIP logit_scale parameter
- run util.run_clip_training with ProbeContext to inject the CLIP loss
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import training_utils as tu
import util as u
from moco_training import resolve_checkpoint_dir, resolve_path
from tracking import ProbeContext, TrainingProbe


class _NullExperiment:
    def log_parameters(self, *_args, **_kwargs) -> None:
        return

    def log_metric(self, *_args, **_kwargs) -> None:
        return

    def end(self) -> None:
        return


def _try_create_experiment(name: str):
    try:
        from comet_utils import create_comet_experiment  # type: ignore

        return create_comet_experiment(name)
    except Exception:
        return _NullExperiment()


def _resolve_data_root(config_path: Path, candidate: Optional[Path | str]) -> Path:
    resolved = resolve_path(config_path.parent, candidate)
    if resolved is not None:
        return resolved
    return (config_path.parent.parent / "data").resolve()


def _build_loaders(
    config_path: Path, data_cfg: Dict[str, object], *, seed: int
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, object]]:
    dataset_type = str(data_cfg.get("dataset_type", "cronos")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))

    data_root = _resolve_data_root(config_path, data_cfg.get("data_dir"))
    normalize = bool(data_cfg.get("normalize", True))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio_cfg = data_cfg.get("val_ratio")
    val_ratio = float(val_ratio_cfg) if val_ratio_cfg is not None else 0.2
    if dataset_type == "cronos" and val_ratio <= 0.0:
        val_ratio = 0.2

    cronos_kwargs: Dict[str, object] = dict(data_cfg.get("cronos_kwargs", {}) or {})
    datasets_spec = data_cfg.get("datasets")
    dataset_name = data_cfg.get("dataset_name")

    if dataset_type == "cronos":
        # Ensure offline cache defaults for Chronos loads.
        load_kwargs = cronos_kwargs.setdefault("load_kwargs", {})
        if isinstance(load_kwargs, dict):
            load_kwargs.setdefault("offline_cache_dir", str(data_root))
            load_kwargs.setdefault("force_offline", True)

        # Optional: load dataset list from a config yaml if not explicitly provided.
        if not datasets_spec and dataset_name is None:
            candidates: Sequence[object] = (
                data_cfg.get("cronos_config"),
                config_path.parent / "cronos_loader_example.yaml",
            )
            for cand in candidates:
                resolved = resolve_path(config_path.parent, cand) if cand is not None else None
                if resolved is not None and resolved.exists():
                    import yaml  # local import to keep this file lightweight

                    with resolved.open("r", encoding="utf-8") as handle:
                        raw = yaml.safe_load(handle) or {}
                    raw_datasets = raw.get("datasets_to_load")
                    if isinstance(raw_datasets, Sequence) and not isinstance(raw_datasets, (str, bytes)):
                        datasets_spec = [str(x) for x in raw_datasets if str(x).strip()]
                    repo_id = raw.get("repo_id")
                    if repo_id is not None:
                        cronos_kwargs.setdefault("repo_id", repo_id)
                    break

    train_loader, val_loader = u.build_time_series_dataloaders(
        data_dir=str(data_root),
        filename=data_cfg.get("filename"),
        dataset_name=dataset_name,
        datasets=datasets_spec,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        dataset_type=dataset_type,
        val_split=data_cfg.get("val_split"),
        seed=seed,
        cronos_kwargs=cronos_kwargs,
    )

    meta = {
        "dataset_type": dataset_type,
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "normalize": normalize,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "data_root": str(data_root),
        "datasets": list(datasets_spec)
        if isinstance(datasets_spec, Sequence) and not isinstance(datasets_spec, (str, bytes))
        else datasets_spec,
    }
    return train_loader, val_loader, meta


def _infer_feature_dim(loader: DataLoader) -> Tuple[int, int]:
    it = iter(loader)
    try:
        batch = next(it)
    except StopIteration as exc:
        raise ValueError("Training data loader produced no batches.") from exc
    seq = u.prepare_sequence(u.extract_sequence(batch))
    return int(seq.shape[-1]), int(seq.shape[1])


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified CLIP-style training (based on cosine_training.py).")
    default_cfg = Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Directory containing *_last.pt files.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None, help="Max grad norm (0 disables).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    experiment = _try_create_experiment("clip_training")

    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)

    training_cfg = config.training
    epochs = int(args.epochs) if args.epochs is not None else int(training_cfg.get("epochs", 100))
    grad_clip_cfg = training_cfg.get("grad_clip", 1.0) if args.grad_clip is None else args.grad_clip
    max_grad_norm = None if grad_clip_cfg is None or float(grad_clip_cfg) <= 0.0 else float(grad_clip_cfg)

    init_temp = float(args.temperature) if args.temperature is not None else float(training_cfg.get("temperature", 0.07))
    if init_temp <= 0:
        raise ValueError("temperature must be > 0")

    experiment.log_parameters(
        {
            "seed": config.seed,
            "config_file": str(config_path),
            "device": str(device),
            "epochs": epochs,
            "temperature_init": init_temp,
            "grad_clip": 0.0 if max_grad_norm is None else max_grad_norm,
        }
    )

    train_loader, val_loader, loader_meta = _build_loaders(config_path, config.data, seed=config.seed)
    feat_dim, seq_len = _infer_feature_dim(train_loader)
    experiment.log_parameters({**loader_meta, "feature_dim": feat_dim, "sequence_length": seq_len})

    encoder = tu.build_encoder_from_config(config.model).to(device)
    visual_encoder = tu.build_visual_encoder_from_config(config.model).to(device)

    projection_dim = int(config.model.get("model_dim", 128))
    projection_head = u.build_projection_head(encoder, output_dim=projection_dim).to(device)
    visual_projection_head = u.build_projection_head(visual_encoder, output_dim=projection_dim).to(device)

    # Learnable CLIP logit scale (leaf parameter on correct device).
    logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp), dtype=torch.float32, device=device))

    base_lr = float(training_cfg.get("learning_rate", 1e-3))
    enc_lr = float(training_cfg.get("encoder_lr", base_lr))
    head_lr = float(training_cfg.get("head_lr", base_lr))
    logit_lr = float(training_cfg.get("logit_scale_lr", head_lr))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    optimizer = torch.optim.AdamW(
        [
            {"params": list(encoder.parameters()), "lr": enc_lr, "weight_decay": weight_decay},
            {"params": list(visual_encoder.parameters()), "lr": enc_lr, "weight_decay": weight_decay},
            {"params": list(projection_head.parameters()), "lr": head_lr, "weight_decay": weight_decay},
            {"params": list(visual_projection_head.parameters()), "lr": head_lr, "weight_decay": weight_decay},
            {"params": [logit_scale], "lr": logit_lr, "weight_decay": 0.0},
        ],
        lr=base_lr,
        weight_decay=weight_decay,
    )

    initial_epoch = 0
    best_loss: Optional[float] = None
    resume_dir: Optional[Path] = None

    if args.resume_checkpoint is not None:
        resume_candidate = resolve_path(Path.cwd(), args.resume_checkpoint)
        if resume_candidate is None:
            raise FileNotFoundError(f"Unable to resolve resume path: {args.resume_checkpoint}")
        resume_dir = resume_candidate if resume_candidate.is_dir() else resume_candidate.parent

        def _load_component(name: str, module: nn.Module) -> dict:
            path = resume_dir / f"{name}_last.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint file: {path}")
            state = torch.load(path, map_location="cpu")
            module.load_state_dict(state["model_state_dict"])
            return state

        state_ts = _load_component("time_series", encoder)
        _load_component("visual_encoder", visual_encoder)
        _load_component("time_series_projection", projection_head)
        _load_component("visual_projection", visual_projection_head)

        optimizer.load_state_dict(state_ts["optimizer_state_dict"])
        stored_epoch = int(state_ts.get("epoch", 0))
        initial_epoch = min(epochs, stored_epoch + 1)

        loss_candidate = state_ts.get("loss")
        if loss_candidate is not None:
            best_loss = float(loss_candidate)

    checkpoint_dir = (
        resume_dir.resolve() if resume_dir is not None else resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    )

    def clip_loss(xq: torch.Tensor, xk: torch.Tensor, _temperature: float) -> torch.Tensor:
        xq = F.normalize(xq, dim=1)
        xk = F.normalize(xk, dim=1)
        scale = logit_scale.clamp(max=math.log(100.0)).exp()
        logits = scale * (xq @ xk.t())
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    probe = TrainingProbe(
        experiment=experiment,
        noise_std=float(training_cfg.get("noise_std", 0.01)),
        min_grad_norm=float(training_cfg.get("min_grad_norm", 0.0)),
        max_grad_norm=max_grad_norm,
    )

    tracked_params = [
        ("encoder.output_proj.weight", encoder.output_proj.weight),
        ("visual_encoder.output_proj.weight", visual_encoder.output_proj.weight),
        ("logit_scale", logit_scale),
    ]

    if val_loader is not None and len(val_loader) == 0:
        val_loader = None

    with ProbeContext(
        u_module=u,
        optimizer=optimizer,
        probe=probe,
        tracked_params=tracked_params,
        loss_impl=clip_loss,
    ):
        u.run_clip_training(
            encoder=encoder,
            visual_encoder=visual_encoder,
            projection_head=projection_head,
            visual_projection_head=visual_projection_head,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            checkpoint_dir=checkpoint_dir,
            epochs=epochs,
            noise_std=float(training_cfg.get("noise_std", 0.01)),
            optimizer=optimizer,
            initial_epoch=initial_epoch,
            best_loss=best_loss,
            experiment=experiment,
        )

    experiment.end()


if __name__ == "__main__":
    main()