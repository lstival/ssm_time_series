"""MoCo training entry-point that feeds on Chronos datasets."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, Optional

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
    
    def __init__(self, encoder: nn.Module, input_dim: int, queue_size: int = 1280, momentum: float = 0.999, temperature: float = 0.07):
        super().__init__()
        
        # Create query encoder
        self.encoder_q = encoder
        
        # Create momentum key encoder
        self.encoder_k = copy.deepcopy(encoder)
        
        # Disable gradients for key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        min_seq_len = getattr(self.encoder_q, "token_size", None)
        if min_seq_len is None:
            min_seq_len = getattr(self.encoder_q, "patch_length", None)
        if not isinstance(min_seq_len, int) or min_seq_len <= 0:
            min_seq_len = 1
        self.min_seq_len = min_seq_len

        # Infer encoder output dimension by doing a forward pass
        encoder_output_dim = self.encoder_k.final_norm._parameters["weight"].shape[0]
        
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
            
        # Query forward pass
        q_encoded = self.encoder_q(x_q)
        q = self.projection_head_q(q_encoded)
        q = nn.functional.normalize(q, dim=1)
        
        # Key forward pass (no gradients)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k_encoded = self.encoder_k(x_k)
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

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """Exponential moving average for encoder and projection head."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)

        for buffer_q, buffer_k in zip(self.encoder_q.buffers(), self.encoder_k.buffers()):
            buffer_k.data.copy_(buffer_q.data)

        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)

        for buffer_q, buffer_k in zip(self.projection_head_q.buffers(), self.projection_head_k.buffers()):
            buffer_k.data.copy_(buffer_q.data)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Maintain the negative sample queue used for contrastive training."""
        keys = nn.functional.normalize(keys, dim=1)
        batch_size = keys.shape[0]
        if batch_size == 0:
            return

        ptr = int(self.queue_ptr.item())
        keys_t = keys.transpose(0, 1).contiguous()

        if batch_size >= self.queue_size:
            keys_t = keys_t[:, -self.queue_size :]
            batch_size = keys_t.shape[1]

        end_ptr = ptr + batch_size
        if end_ptr <= self.queue_size:
            self.queue[:, ptr:end_ptr] = keys_t
            ptr = end_ptr % self.queue_size
        else:
            first_part = self.queue_size - ptr
            self.queue[:, ptr:] = keys_t[:, :first_part]
            remaining = batch_size - first_part
            if remaining > 0:
                self.queue[:, :remaining] = keys_t[:, first_part:first_part + remaining]
            ptr = remaining % self.queue_size

        self.queue_ptr[0] = ptr


def resolve_path(base: Path, candidate: Optional[Path | str]) -> Optional[Path]:
    if candidate is None:
        return None
    candidate = Path(candidate).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def prepare_dataset(
    config_path: Path,
    config_data: dict,
) -> Dataset:
    cronos_config = config_data.get("cronos_config")
    if cronos_config is None:
        cronos_config = config_path.parent / "cronos_loader_example.yaml"
    cronos_config = resolve_path(config_path.parent, cronos_config)
    if cronos_config is None or not cronos_config.exists():
        raise FileNotFoundError(f"Cronos loader config not found: {cronos_config}")

    split = config_data.get("split")
    patch_length = config_data.get("patch_length")
    load_kwargs = dict(config_data.get("load_kwargs", {}) or {})
    normalize = config_data.get("normalize", True)

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
        normalize=normalize,
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
    scaler = torch.amp.GradScaler(enabled=use_amp, device=device)
    autocast_device = device.type if device.type in {"cuda", "cpu", "hip", "xpu"} else "cuda"
    
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
            x_q = seq.swapaxes(1,2)
            x_k = seq.swapaxes(1,2)

            with torch.amp.autocast(device_type=autocast_device, enabled=scaler.is_enabled()):
                logits, labels, loss = model(x_q, x_k)

            scaler.scale(loss).backward()

            if max_grad_norm is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            
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
                    x_q = seq.swapaxes(1,2)
                    x_k = seq.swapaxes(1,2)
                    
                    with torch.amp.autocast(device_type=autocast_device, enabled=scaler.is_enabled()):
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

    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=config.seed)

    batch_size = int(data_cfg.get("batch_size", 128))
    val_batch_size = int(data_cfg.get("val_batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    # Ensure batch size is compatible with queue size
    queue_size = int(config.training.get("queue_size", 65536))
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
    temperature = float(config.training.get("temperature", 0.07))
    momentum = float(config.training.get("momentum", 0.999))
    
    model = MoCoModel(
        encoder=encoder,
        input_dim=feature_dim,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature
    )
    
    optimizer = tu.build_optimizer(model, config.training)

    epochs = int(config.training.get("epochs", 100))
    scheduler = tu.build_scheduler(optimizer, config.training, epochs)

    checkpoint_dir = resolve_checkpoint_dir(config, config_path, None)
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