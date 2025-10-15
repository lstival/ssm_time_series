from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import torch.nn as nn

import training_utils as tu
import util as u
from models.mamba_encoder import MambaEncoder


class ContrastiveModel(nn.Module):
    """Wrap the encoder so any input feature dimension is accepted."""

    def __init__(self, encoder: MambaEncoder, input_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter: nn.Module = (
            nn.Linear(input_dim, encoder.input_dim, bias=False)
            if input_dim != encoder.input_dim
            else nn.Identity()
        )

    def forward(self, x):
        return self.encoder(self.adapter(x))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Contrastive training for the Mamba encoder")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--filename", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--val_batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--dataset_type", type=str, default=None)
    p.add_argument("--datasets", type=str, nargs="*", default=None)
    p.add_argument("--val_split", type=float, default=None)
    return p.parse_args(list(argv) if argv is not None else None)


def resolve_config_path(cfg_arg: Path) -> Path:
    cfg_path = cfg_arg if cfg_arg.is_absolute() else (Path(__file__).resolve().parent / cfg_arg).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    return cfg_path


def prepare_data_loaders(config, args, root_dir):
    data_cfg = config.data
    data_dir_arg = args.data_dir if args.data_dir is not None else data_cfg.get("data_dir", "")
    data_dir = Path(data_dir_arg)
    if not data_dir.is_absolute():
        data_dir = (root_dir / data_dir).resolve()

    dataset_type = (args.dataset_type or str(data_cfg.get("dataset_type", "icml"))).lower()
    dataset_name = args.dataset_name if args.dataset_name is not None else data_cfg.get("dataset_name")
    filename = args.filename if args.filename is not None else data_cfg.get("filename")
    batch_size = args.batch_size if args.batch_size is not None else int(data_cfg.get("batch_size", 128))
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else int(data_cfg.get("val_batch_size", 256))
    num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    normalize = bool(data_cfg.get("normalize", True))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    datasets = args.datasets if args.datasets else data_cfg.get("datasets")
    val_split = args.val_split if args.val_split is not None else data_cfg.get("val_split")
    cronos_opts = data_cfg.get("cronos", {}) or {}

    train_loader, val_loader = u.build_time_series_dataloaders(
        data_dir=data_dir,
        filename=filename,
        dataset_name=dataset_name,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        dataset_type=dataset_type,
        datasets=datasets,
        val_split=val_split,
        seed=config.seed,
        cronos_kwargs=cronos_opts,
    )

    sample_batch = next(iter(train_loader))
    sample_seq = u.prepare_sequence(u.extract_sequence(sample_batch))
    feature_dim = sample_seq.shape[-1]

    return train_loader, val_loader, feature_dim


def build_model_and_optim(config, feature_dim, args):
    encoder = tu.build_encoder_from_config(config.model)
    model = ContrastiveModel(encoder, input_dim=feature_dim)

    training_cfg = config.training
    epochs = int(args.epochs) if args.epochs is not None else int(training_cfg.get("epochs", 100))
    temperature = float(args.temperature) if args.temperature is not None else float(training_cfg.get("temperature", 0.2))
    optimizer = tu.build_optimizer(model, training_cfg)
    scheduler = tu.build_scheduler(optimizer, training_cfg, epochs)

    return model, optimizer, scheduler, epochs, temperature


def prepare_checkpoint_dir(config, config_path: Path, args):
    logging_cfg = config.logging
    root = args.checkpoint_dir if args.checkpoint_dir is not None else Path(logging_cfg.get("checkpoint_dir", "./checkpoints"))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()

    # Create and return a run-specific checkpoint directory (avoid calling missing util function)
    run_dir = root / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = tu.load_config(config_path)

    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    root_dir = config_path.parent
    train_loader, val_loader, feature_dim = prepare_data_loaders(config, args, root_dir)

    model, optimizer, scheduler, epochs, temperature = build_model_and_optim(config, feature_dim, args)
    checkpoint_dir = prepare_checkpoint_dir(config, config_path, args)
    print(f"Checkpoints: {checkpoint_dir}")

    val_loader = val_loader if val_loader is not None and len(val_loader) > 0 else None

    u.run_contrastive_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        temperature=temperature,
        device=device,
        use_amp=bool(config.training.get("use_amp", False)),
        max_grad_norm=float(config.training.get("max_grad_norm", 0.0)) or None,
        checkpoint_dir=checkpoint_dir,
        save_best_only=bool(config.logging.get("save_best_only", True)),
        save_last=bool(config.logging.get("save_last", True)),
    )
