"""Simple forecasting training script for MambaVisualEncoder."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.mamba_visual_encoder import MambaVisualEncoder
from util import build_time_series_dataloaders, default_device, prepare_run_directory


class ForecastModel(nn.Module):
    """Forecasting model using MambaVisualEncoder as backbone."""
    
    def __init__(
        self,
        encoder: MambaVisualEncoder,
        forecast_len: int = 192,
    ):
        super().__init__()
        self.encoder = encoder
        self.forecast_len = forecast_len
        # Project from embedding_dim to forecast_len
        self.forecast_head = nn.Linear(encoder.embedding_dim, forecast_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and forecast head.
        
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        
        Returns:
            Forecasts of shape (batch, forecast_len)
        """
        embeddings = self.encoder(x)
        return self.forecast_head(embeddings)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # batch = (seq_x, seq_y, seq_x_mark, seq_y_mark)
        # seq_x: (batch, seq_len, features)
        # seq_y: (batch, label_len + pred_len, features)
        seq_x, seq_y, _, _ = batch
        
        # Move to device and transpose for encoder (expects batch, channels, seq_len)
        seq_x = seq_x.to(device).float().transpose(1, 2)  # (batch, features, seq_len)
        seq_y = seq_y.to(device).float()  # (batch, label_len + pred_len, features)
        
        # Extract forecast target - flatten to (batch, total_timesteps)
        # Assuming we want to predict all features across all future timesteps
        target = seq_y.reshape(seq_y.size(0), -1)  # (batch, forecast_len)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(seq_x)
        loss = criterion(predictions, target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            seq_x, seq_y, _, _ = batch
            
            seq_x = seq_x.to(device).float().transpose(1, 2)
            seq_y = seq_y.to(device).float()
            
            target = seq_y.reshape(seq_y.size(0), -1)
            
            predictions = model(seq_x)
            loss = criterion(predictions, target)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train MambaVisualEncoder for forecasting")
    parser.add_argument("--data_dir", type=str, default="../ICML_datasets",
                        help="Path to dataset directory")
    parser.add_argument("--filename", type=str, default=None,
                        help="Name for use a single dataset from 'data_dir/filename'")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="Specific dataset name (empty for all datasets)")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=12,
                        help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--token_size", type=int, default=32,
                        help="Token size for encoder")
    parser.add_argument("--model_dim", type=int, default=256,
                        help="Model dimension")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Output embedding dimension")
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of Mamba blocks")
    parser.add_argument("--forecast_len", type=int, default=192,
                        help="Forecast output length (should match label_len + pred_len * features)")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Setup device
    device = default_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Setting up data loaders...")
    train_loader, val_loader = build_time_series_dataloaders(
        data_dir=args.data_dir,
        filename=args.filename,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Get a sample to determine actual forecast_len
    sample_batch = next(iter(train_loader))
    seq_x_sample, seq_y_sample = sample_batch[0], sample_batch[1]
    actual_forecast_len = seq_y_sample.size(1) * seq_y_sample.size(2)
    print(f"Sample shapes - seq_x: {seq_x_sample.shape}, seq_y: {seq_y_sample.shape}")
    print(f"Actual forecast length: {actual_forecast_len}")
    
    # Override forecast_len with actual value
    args.forecast_len = actual_forecast_len
    
    # Create model
    print("Building model...")
    encoder = MambaVisualEncoder(
        input_dim=args.token_size,
        model_dim=args.model_dim,
        depth=args.depth,
        embedding_dim=args.embedding_dim,
        pooling="mean",
        dropout=0.1,
    )
    
    model = ForecastModel(
        encoder=encoder,
        forecast_len=args.forecast_len,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Create checkpoint directory
    run_dir = prepare_run_directory(Path(args.checkpoint_dir), "forecast_encoder")
    print(f"Checkpoint directory: {run_dir}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = run_dir / "best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args),
            }, checkpoint_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save last checkpoint
        last_checkpoint_path = run_dir / "last.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args),
        }, last_checkpoint_path)
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {run_dir}")


if __name__ == "__main__":
    main()
