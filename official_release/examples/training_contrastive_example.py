import torch
from torch.utils.data import DataLoader
from cm_mamba.models.mamba_encoder import MambaEncoder
from cm_mamba.training.loops import forward_contrastive_batch, info_nce_loss
from cm_mamba.data.utils import create_contrastive_views

def contrastive_training_demo():
    # 1. Setup Model
    model = MambaEncoder(input_dim=32, model_dim=128, embedding_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 2. Dummy Batch (Batch, Time, Features)
    batch = torch.randn(8, 96, 32).to(device)
    
    # 3. Training step simulation
    model.train()
    optimizer.zero_grad()
    
    # Create two augmented views
    x1, x2 = create_contrastive_views(batch)
    
    # Forward pass
    z1 = model(x1)
    z2 = model(x2)
    
    # Calculate Contrastive Loss (InfoNCE)
    loss = info_nce_loss(z1, z2, temperature=0.07)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"Contrastive Loss: {loss.item():.4f}")

if __name__ == "__main__":
    contrastive_training_demo()
