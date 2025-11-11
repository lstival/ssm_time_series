"""Cache encoder embeddings for ICML datasets with multiple forecast horizons.

This utility loads the pretrained encoder defined by the standard configuration,
iterates over every ICML dataset split, extracts embeddings batch-by-batch, and
persists them to disk without keeping the entire cache in GPU memory. Each saved
file mirrors the original batch size and contains the encoder embedding together
with multiple forecast targets (96/192/336/720 steps).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

import training_utils as tu
from moco_training import resolve_path
from time_series_loader import TimeSeriesDataModule
from util import default_device

@dataclass
class Args:
    config: str = str(SRC_DIR / "configs" / "mamba_encoder.yaml")
    encoder_checkpoint: str = str(ROOT_DIR / "checkpoints" / "ts_encoder_20251101_1100" / "time_series_best.pt")
    data_dir: Optional[str] = None
    output_dir: str = str(ROOT_DIR / "embedding_cache")
    dataset_name: Optional[str] = None
    filename: Optional[str] = None
    horizons: str = "96,192,336,720"
    num_workers: Optional[int] = None
    device: str = "auto"
    seed: Optional[int] = None
    force: bool = False

@dataclass
class SplitSummary:
    samples: int
    batches: int
    files: List[Path]


def _parse_horizons(raw: str) -> List[int]:
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    horizons = sorted({int(value) for value in parts})
    if not horizons:
        raise ValueError("At least one forecast horizon must be provided")
    expected = {96, 192, 336, 720}
    missing = [value for value in horizons if value not in expected]
    if missing:
        raise ValueError(f"Unsupported horizons requested: {missing}; expected subset of {sorted(expected)}")
    return horizons


def _ensure_dataset_pred_len(loader: Optional[DataLoader], pred_len: int) -> None:
    if loader is None:
        return

    def _apply(obj: object) -> None:
        if obj is None:
            return
        if isinstance(obj, ConcatDataset):
            for child in obj.datasets:
                _apply(child)
            return
        if hasattr(obj, "dataset"):
            _apply(getattr(obj, "dataset"))
        if hasattr(obj, "pred_len"):
            setattr(obj, "pred_len", pred_len)

    _apply(loader.dataset)


def _build_module(
    config_path: Path,
    config: tu.ExperimentConfig,
    args: Args,
    *,
    device: torch.device,
) -> Tuple[TimeSeriesDataModule, Dict[str, object]]:
    data_cfg = dict(config.data or {})

    base_dir = config_path.parent
    default_data_dir = data_cfg.get("data_dir")
    if default_data_dir is not None:
        candidate = Path(default_data_dir)
        if not candidate.is_absolute():
            default_data_dir = str((base_dir / candidate).resolve())
        else:
            default_data_dir = str(candidate)
    else:
        default_data_dir = str((ROOT_DIR / "ICML_datasets").resolve())

    data_dir_input = args.data_dir or default_data_dir
    data_dir_path = resolve_path(base_dir, data_dir_input)
    data_dir_final = str(data_dir_path) if data_dir_path is not None else data_dir_input

    filename = args.filename if args.filename is not None else data_cfg.get("filename")
    if filename is not None:
        filename_path = resolve_path(base_dir, filename)
        if filename_path is not None:
            filename = str(filename_path)

    dataset_name = args.dataset_name if args.dataset_name is not None else data_cfg.get("dataset_name", "")
    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 4))
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))

    module = TimeSeriesDataModule(
        dataset_name=dataset_name or "",
        data_dir=data_dir_final,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        normalize=bool(data_cfg.get("normalize", True)),
        filename=filename,
        train=True,
        val=True,
        test=True,
    )

    module.setup()
    return module, {
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "data_dir": data_dir_final,
        "dataset_name": dataset_name,
        "filename": filename,
    }


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


def _save_batch(
    output_dir: Path,
    batch_index: int,
    *,
    embeddings: torch.Tensor,
    targets: Dict[int, torch.Tensor],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"embeddings": embeddings.cpu()}
    for horizon, tensor in targets.items():
        payload[f"targets_{horizon}"] = tensor.cpu()
    target_path = output_dir / f"batch_{batch_index:05d}.pt"
    torch.save(payload, target_path)
    return target_path


def _extract_split(
    encoder: nn.Module,
    loader: Optional[DataLoader],
    *,
    device: torch.device,
    horizons: Sequence[int],
    max_horizon: int,
    output_dir: Path,
) -> SplitSummary:
    if loader is None:
        return SplitSummary(samples=0, batches=0, files=[])

    _ensure_dataset_pred_len(loader, max_horizon)

    stored_files: List[Path] = []
    total_samples = 0
    encoder.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"Extract {output_dir.name}", leave=False)):
            seq_x = batch[0].to(device).float().transpose(1, 2)
            seq_y = batch[1].float()

            if seq_x.shape[1] > 1:
                # outs = []
                # for feat_idx in range(seq_x.shape[1]):
                #     # pass each feature/channel separately through the encoder
                #     emb = encoder(seq_x[:, feat_idx, :].unsqueeze(1))
                #     outs.append(emb)
                # # outs is list of tensors with shape (B, E, ...)
                # stacked = torch.stack(outs, dim=-1)  # (B, F, E, ...)
                # # flatten feature+embedding dims into (B, F * E * ...)
                # embeddings = stacked.swapaxes(1,2)
                embeddings = encoder(seq_x[:,:1,:])
            else:
                # single feature channel: run directly
                embeddings = encoder(seq_x)

            # embeddings = encoder(seq_x).detach().cpu()
            seq_y_forecast = seq_y[:, -max_horizon:, :]
            targets = {h: seq_y_forecast[:, -h:, :] for h in horizons}

            file_path = _save_batch(output_dir, batch_index, embeddings=embeddings, targets=targets)

            stored_files.append(file_path)
            total_samples += embeddings.size(0)

            del seq_x, seq_y, embeddings, seq_y_forecast

    return SplitSummary(samples=total_samples, batches=len(stored_files), files=stored_files)


def _write_metadata(
    root: Path,
    *,
    dataset_name: str,
    split_summaries: Dict[str, SplitSummary],
    horizons: Sequence[int],
    encoder_dim: int,
    meta: Dict[str, object],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset_name,
        "horizons": list(horizons),
        "embedding_dim": int(encoder_dim),
        "splits": {
            split: {
                "samples": summary.samples,
                "batches": summary.batches,
                "files": [str(path.name) for path in summary.files],
            }
            for split, summary in split_summaries.items()
            if summary.batches > 0
        },
        "data_cfg": meta,
    }
    with (root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _dataset_already_cached(dataset_dir: Path, requested_horizons: Sequence[int]) -> bool:
    """Return True if embeddings for this dataset already exist.

    Heuristic:
    - If metadata.json exists and its horizons cover the requested horizons AND
      it reports at least one batch in any split, consider it cached.
    - Otherwise, if any batch_*.pt exists in train/val/test subdirs, consider it cached.
    """
    if not dataset_dir.exists():
        return False

    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            stored_horizons = set(int(h) for h in meta.get("horizons", []))
            if all(int(h) in stored_horizons for h in requested_horizons):
                splits = meta.get("splits") or {}
                if any(int(s.get("batches", 0)) > 0 for s in splits.values()):
                    return True
        except Exception:
            # Fall back to file presence checks
            pass

    # Fallback: presence of any saved batch files under known splits
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if split_dir.exists() and any(split_dir.glob("batch_*.pt")):
            return True
    return False


def main(args: Optional[Args] = None) -> None:
    args = args or Args()
    config_path = resolve_path(Path.cwd(), Path(args.config))
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = tu.load_config(config_path)

    seed = args.seed if args.seed is not None else config.seed
    tu.set_seed(seed)

    device = default_device() if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")

    horizons = _parse_horizons(args.horizons)
    max_horizon = max(horizons)

    module, data_meta = _build_module(config_path, config, args, device=device)
    dataset_groups = module.get_dataloaders()
    if not dataset_groups:
        raise RuntimeError("No datasets discovered for extraction")

    checkpoint_path = resolve_path(config_path.parent, Path(args.encoder_checkpoint))
    if checkpoint_path is None:
        checkpoint_path = Path(args.encoder_checkpoint).expanduser().resolve()

    encoder = _load_encoder(config, checkpoint_path, device=device)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Caching embeddings to {output_root}")

    for group in dataset_groups:
        dataset_dir = output_root / group.name
        # Skip dataset if embeddings already exist (unless forced)
        if not args.force and _dataset_already_cached(dataset_dir, horizons):
            print(f"\nSkipping dataset: {group.name} (existing cache found at {dataset_dir}). Use --force to overwrite.")
            continue
        summaries: Dict[str, SplitSummary] = {}
        print(f"\nProcessing dataset: {group.name}")
        for split_name, loader in ("train", group.train), ("val", group.val), ("test", group.test):
            if loader is None:
                continue
            split_dir = dataset_dir / split_name
            summary = _extract_split(
                encoder,
                loader,
                device=device,
                horizons=horizons,
                max_horizon=max_horizon,
                output_dir=split_dir,
            )
            summaries[split_name] = summary
            print(f"  {split_name}: {summary.samples} samples across {summary.batches} batches")

        if summaries:
            encoder_dim = getattr(encoder, "embedding_dim", None)
            if encoder_dim is None:
                dummy_seq = torch.zeros(1, getattr(encoder, "input_dim", 1), getattr(encoder, "input_dim", 1), device=device)
                with torch.no_grad():
                    encoder_dim = int(encoder(dummy_seq).shape[-1])
            else:
                encoder_dim = int(encoder_dim)
            _write_metadata(
                dataset_dir,
                dataset_name=group.name,
                split_summaries=summaries,
                horizons=horizons,
                encoder_dim=encoder_dim,
                meta=data_meta,
            )

    print("\nFinished caching embeddings.")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
