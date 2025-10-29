"""MoCo training entry-point that feeds on Chronos datasets."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import training_utils as tu
import util as u
from dataloaders.cronos_loader_ts import load_cronos_time_series_dataset
from tqdm import tqdm


class MoCoProjectionHead(nn.Module):
    """Projection head for MoCo model."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MoCoModel(nn.Module):
    """MoCo model with momentum-updated key encoder and queue."""
    
    def __init__(self, encoder: nn.Module, input_dim: int, queue_size: int = 65536, momentum: float = 0.999, temperature: float = 0.07):
        super().__init__()
        
        # Create query encoder with adapter if needed
        self.encoder_q = encoder
        # For 3D inputs, we need to apply the linear layer to the last dimension
        if hasattr(encoder, "input_dim") and input_dim != encoder.input_dim:
            self.adapter_q = nn.Linear(input_dim, encoder.input_dim, bias=False)
        else:
            self.adapter_q = nn.Identity()
        
        # Create momentum key encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.adapter_k = copy.deepcopy(self.adapter_q)
        
        # Disable gradients for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.adapter_k.parameters():
            param.requires_grad = False
            
        # Infer encoder output dimension by doing a forward pass
        encoder_output_dim = self._infer_encoder_output_dim(input_dim)
        
        # Projection heads
        self.projection_head_q = MoCoProjectionHead(encoder_output_dim, encoder_output_dim, 128)
        self.projection_head_k = copy.deepcopy(self.projection_head_q)
        
        # Disable gradients for key projection head
        for param in self.projection_head_k.parameters():
            param.requires_grad = False
            
        # MoCo parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Initialize queue
        self.register_buffer("queue", torch.randn(128, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _infer_encoder_output_dim(self, input_dim: int) -> int:
        """Infer the output dimension of the encoder by doing a forward pass."""
        self.encoder_q.eval()
        with torch.no_grad():
            # Create a dummy input with the expected shape (batch, seq, features)
            # Use a reasonable sequence length for inference
            seq_length = 64  # Adjust based on your typical sequence length
            dummy_input = torch.randn(1, seq_length, input_dim)
            adapted_input = self.adapter_q(dummy_input)
            output = self.encoder_q(adapted_input)
            return output.shape[-1]
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
        
        for param_q, param_k in zip(self.adapter_q.parameters(), self.adapter_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys."""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, "Queue size should be divisible by batch size"
        
        # Replace keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr
    
    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through MoCo model.
        
        Args:
            x_q: Query samples - shape (batch, seq, features) or (batch, features)
            x_k: Key samples (positive keys) - shape (batch, seq, features) or (batch, features)
            
        Returns:
            logits: Logits for contrastive loss
            targets: Target labels (zeros for positive pairs)
            loss: InfoNCE loss
        """
        # Ensure inputs have the right shape (batch, seq, features)
        if x_q.dim() == 2:
            x_q = x_q.unsqueeze(1)  # Add sequence dimension
        if x_k.dim() == 2:
            x_k = x_k.unsqueeze(1)  # Add sequence dimension
            
        # Query forward pass
        q_adapted = self.adapter_q(x_q)
        q_encoded = self.encoder_q(q_adapted)
        q = self.projection_head_q(q_encoded)
        q = nn.functional.normalize(q, dim=1)
        
        # Key forward pass (no gradients)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k_adapted = self.adapter_k(x_k)
            k_encoded = self.encoder_k(k_adapted)
            k = self.projection_head_k(k_encoded)
            k = nn.functional.normalize(k, dim=1)
        
        # Compute logits
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positive key is the 0th
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Compute loss
        loss = nn.functional.cross_entropy(logits, labels)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels, loss


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoCo training using Chronos patched loader")
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
    parser.add_argument("--queue-size", type=int, default=None, help="Size of the negative sample queue")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum for key encoder update")
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
        drop_last=True,  # Important for MoCo queue management
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=True,  # Consistent with training
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


def run_moco_training(
    model: MoCoModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = None,
    checkpoint_dir: Path = Path("./checkpoints"),
    save_best_only: bool = True,
    save_last: bool = True,
) -> None:
    """Run MoCo training loop."""
    
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train", leave=False, unit="batch")
        for batch in progress:
            optimizer.zero_grad()
            
            # Extract sequences and create positive pairs
            seq = u.prepare_sequence(u.extract_sequence(batch)).to(device)
            
            # For MoCo, we need query and key samples
            # Simple approach: use the same sequence as both query and key (self-supervised)
            # In practice, you might want to apply different augmentations
            x_q = seq
            x_k = seq  # You could apply different augmentations here
            
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits, labels, loss = model(x_q, x_k)
                scaler.scale(loss).backward()
            
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, labels, loss = model(x_q, x_k)
                loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1

            # Update tqdm postfix with running averages
            avg_loss = train_loss / num_batches if num_batches > 0 else 0.0
            current_lr = optimizer.param_groups[0]['lr']
            progress.set_postfix({"avg_loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.6f}"})
        
        train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    seq = u.prepare_sequence(u.extract_sequence(batch)).to(device)
                    x_q = seq
                    x_k = seq
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            logits, labels, loss = model(x_q, x_k)
                    else:
                        logits, labels, loss = model(x_q, x_k)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
        
        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loader is not None else train_loss)
            else:
                scheduler.step()
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        if val_loader is not None:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoints
        is_best = val_loss < best_loss if val_loader is not None else train_loss < best_loss
        if is_best:
            best_loss = val_loss if val_loader is not None else train_loss
        
        if save_best_only and is_best:
            checkpoint_path = checkpoint_dir / "best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
        
        if save_last:
            checkpoint_path = checkpoint_dir / "last.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss if val_loader is not None else train_loss,
            }, checkpoint_path)


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

    # Ensure batch size is compatible with queue size
    queue_size = args.queue_size if args.queue_size is not None else int(config.training.get("queue_size", 65536))
    if queue_size % batch_size != 0:
        print(f"Warning: Queue size {queue_size} is not divisible by batch size {batch_size}. Adjusting queue size to {(queue_size // batch_size) * batch_size}")
        queue_size = (queue_size // batch_size) * batch_size

    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    feature_dim = infer_feature_dim(train_loader)

    encoder = tu.build_encoder_from_config(config.model)
    
    # MoCo specific parameters
    temperature = args.temperature if args.temperature is not None else float(config.training.get("temperature", 0.07))
    momentum = args.momentum if args.momentum is not None else float(config.training.get("momentum", 0.999))
    
    model = MoCoModel(
        encoder=encoder,
        input_dim=feature_dim,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature
    )
    
    optimizer = tu.build_optimizer(model, config.training)

    epochs = args.epochs if args.epochs is not None else int(config.training.get("epochs", 100))
    scheduler = tu.build_scheduler(optimizer, config.training, epochs)

    checkpoint_dir = resolve_checkpoint_dir(config, config_path, args.checkpoint_dir)
    print(f"Checkpoints: {checkpoint_dir}")

    val_loader = val_loader if val_loader is not None and len(val_loader) > 0 else None

    run_moco_training(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        device=device,
        use_amp=bool(config.training.get("use_amp", False)),
        max_grad_norm=float(config.training.get("max_grad_norm", 0.0)) or None,
        checkpoint_dir=checkpoint_dir,
        save_best_only=bool(config.logging.get("save_best_only", True)),
        save_last=bool(config.logging.get("save_last", True)),
    )


if __name__ == "__main__":
    main()