"""Contrastive training entry-point that feeds on Chronos datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import training_utils as tu
import util as u
from dataloaders.cronos_loader_ts import load_cronos_time_series_dataset


class ContrastiveModel(nn.Module):
    """Adapter-wrapped encoder that accepts arbitrary input feature sizes."""

    def __init__(self, encoder: nn.Module, input_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter: nn.Module = (
            nn.Linear(input_dim, encoder.input_dim, bias=False)
            if hasattr(encoder, "input_dim") and input_dim != encoder.input_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.adapter(x))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrastive training using Chronos patched loader")
    default_cfg = Path(__file__).resolve().parent / "configs" / "mamba_encoder.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    parser.add_argument("--cronos-config", type=Path, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--patch-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation fraction (0-1).")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", type=int, default=None, choices=[0, 1])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--rp-mode", type=str, default="correct", choices=["correct", "shuffled", "random"], help="RP variant for EXP-2")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--load-kwargs", type=str, nargs="*", default=None,
                        help="Optional key=value overrides forwarded to the dataset loader.")
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_path(base: Path, candidate: Optional[Path | str]) -> Optional[Path]:
    if candidate is None:
        return None
    candidate = Path(candidate).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def parse_key_value_pairs(pairs: Optional[Sequence[str]]) -> dict:
    if not pairs:
        return {}
    result: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, got: {item}")
        key, value = item.split("=", maxsplit=1)
        result[key.strip()] = value.strip()
    return result


def prepare_dataset(
    config_path: Path,
    cronos_config_arg: Optional[Path],
    config_data: dict,
    *,
    split_override: Optional[str],
    patch_length_override: Optional[int],
    load_kwargs_override: Optional[dict],
) -> Dataset:
    cronos_config = cronos_config_arg or config_data.get("cronos_config")
    if cronos_config is None:
        cronos_config = config_path.parent / "cronos_loader_example.yaml"
    cronos_config = resolve_path(config_path.parent, cronos_config)
    if cronos_config is None or not cronos_config.exists():
        raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

    split = split_override or config_data.get("split")
    patch_length = patch_length_override or config_data.get("patch_length")
    load_kwargs = {}
    load_kwargs.update(config_data.get("load_kwargs", {}) or {})
    if load_kwargs_override:
        load_kwargs.update(load_kwargs_override)

    # Set offline cache directory to the local data directory
    data_dir = config_path.parent.parent / "data"
    load_kwargs.setdefault("offline_cache_dir", str(data_dir))
    load_kwargs.setdefault("force_offline", True)
    
    print(f"Using local data directory: {data_dir}")

    dataset = load_cronos_time_series_dataset(
        str(cronos_config),
        split=split,
        patch_length=patch_length,
        load_kwargs=load_kwargs,
    )
    return dataset


def split_dataset(
    dataset: Dataset,
    *,
    val_ratio: float,
    seed: int,
) -> tuple[Dataset, Optional[Dataset]]:
    val_ratio = max(0.0, min(float(val_ratio), 0.9))
    if val_ratio == 0.0 or len(dataset) < 2:
        return dataset, None

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size == 0:
        train_size, val_size = len(dataset) - 1, 1

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    *,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, Optional[DataLoader]]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader


def infer_feature_dim(loader: DataLoader) -> int:
    sample_batch = next(iter(loader))
    sample_seq = u.prepare_sequence(u.extract_sequence(sample_batch))
    return sample_seq.shape[-1]


def resolve_checkpoint_dir(config: tu.ExperimentConfig, cfg_path: Path, override: Optional[Path]) -> Path:
    base_dir = override if override is not None else config.logging.get("checkpoint_dir", "./checkpoints")
    base_dir = resolve_path(cfg_path.parent, base_dir)
    if base_dir is None:
        base_dir = Path("./checkpoints").resolve()
    return u.prepare_run_directory(base_dir, config.experiment_name)


def coalesce_bool(value: Optional[int], default: bool) -> bool:
    if value is None:
        return default
    return bool(int(value))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    config_path = resolve_path(Path.cwd(), args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(config_path)
    tu.set_seed(config.seed)
    device = tu.prepare_device(config.device)
    print(f"Using device: {device}")

    data_cfg = config.data
    dataset = prepare_dataset(
        config_path,
        args.cronos_config,
        data_cfg,
        split_override=args.split,
        patch_length_override=args.patch_length,
        load_kwargs_override=parse_key_value_pairs(args.load_kwargs),
    )

    val_ratio = args.val_ratio if args.val_ratio is not None else float(data_cfg.get("val_ratio", 0.1))
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=config.seed)

    batch_size = args.batch_size if args.batch_size is not None else int(data_cfg.get("batch_size", 128))
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else int(data_cfg.get("val_batch_size", 256))
    num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 0))
    pin_memory = coalesce_bool(args.pin_memory, bool(data_cfg.get("pin_memory", False)))

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    feature_dim = infer_feature_dim(train_loader)

    # Use build_visual_encoder_from_config as it supports RP modes for EXP-2
    encoder = tu.build_visual_encoder_from_config(config.model, rp_mode=args.rp_mode)
    model = ContrastiveModel(encoder, input_dim=feature_dim)
    optimizer = tu.build_optimizer(model, config.training)

    epochs = args.epochs if args.epochs is not None else int(config.training.get("epochs", 100))
    temperature = args.temperature if args.temperature is not None else float(config.training.get("temperature", 0.2))

    scheduler = tu.build_scheduler(optimizer, config.training, epochs)

    checkpoint_dir = resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
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


if __name__ == "__main__":
    main()
