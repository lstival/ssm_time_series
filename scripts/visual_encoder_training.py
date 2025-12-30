"""Simple regression training script for MambaVisualEncoder."""

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from ssm_time_series.models.mamba_visual_encoder import MambaVisualEncoder
from ssm_time_series.data.loader import TimeSeriesDataModule


class RegressionModel(nn.Module):
    """Simple regression model using MambaVisualEncoder as backbone."""
    
    def __init__(
        self,
        encoder: MambaVisualEncoder,
        output_dim: int = 1,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and regression head.
        
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
        
        Returns:
            Predictions of shape (batch, output_dim)
        """
        embeddings = self.encoder(x)
        return self.head(embeddings)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Assuming batch is a tuple (data, target) or just data
        if isinstance(batch, (list, tuple)):
            data = batch[0].to(device).float()
            # For regression, we'll predict the mean of the last timestep
            target = data[:, :, -1].mean(dim=1, keepdim=True)
        else:
            data = batch.to(device).float()
            target = data[:, :, -1].mean(dim=1, keepdim=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(data)
        loss = criterion(predictions, target)
        
        # Backward pass
        loss.backward()
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
            if isinstance(batch, (list, tuple)):
                data = batch[0].to(device).float()
                target = data[:, :, -1].mean(dim=1, keepdim=True)
            else:
                data = batch.to(device).float()
                target = data[:, :, -1].mean(dim=1, keepdim=True)
            
            predictions = model(data)
            loss = criterion(predictions, target)
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train MambaVisualEncoder for regression")
    parser.add_argument("--data_dir", type=str, default="../ICML_datasets",
                        help="Path to dataset directory")
    parser.add_argument("--dataset_name", type=str, default="",
                        help="Specific dataset name (empty for all datasets)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--token_size", type=int, default=32,
                        help="Token size for encoder")
    parser.add_argument("--model_dim", type=int, default=768,
                        help="Model dimension")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Output embedding dimension")
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of Mamba blocks")
    parser.add_argument("--output_dim", type=int, default=1,
                        help="Regression output dimension")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data module
    print("Setting up data loaders...")
    data_module = TimeSeriesDataModule(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        normalize=True,
        train_ratio=0.8,
        val_ratio=0.2,
        train=True,
        val=True,
        test=False,
    )
    
    train_loader, val_loader = data_module.get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
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
    
    model = RegressionModel(
        encoder=encoder,
        output_dim=args.output_dim,
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
    checkpoint_dir = Path(args.checkpoint_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = checkpoint_dir / f"visual_encoder_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
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
