import torch
import torch.nn as nn

from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)

import pytorch_lightning as pl
from training_utils import (
    build_encoder_from_config,
    build_optimizer,
    build_scheduler,
    infer_feature_dim,
    load_config,
    prepare_dataloaders,
    prepare_device,
    set_seed,
)
import os 
from pathlib import Path
import copy
from tqdm.auto import tqdm


## Config File (with the training details)
config_file = Path("configs") / "mamba_encoder.yaml"
config_file = config_file.resolve()
if not config_file.exists():
    raise FileNotFoundError(f"Config file not found: {config_file}")
config = load_config(config_file)

## Loading the dataset
root_dir = config_file.parent
train_loader, val_loader = prepare_dataloaders(config, root_dir)

## MoCo training
memory_bank_size = 1000
max_epochs = 100

device = prepare_device(config.device)
set_seed(config.seed)

# enable medium precision matmul for float32 (PyTorch >= 2.0)
try:
    torch.set_float32_matmul_precision('medium')
    print("Precision defined as Medium")
except AttributeError:
    # no-op on older PyTorch versions
    print("Precision AS NOT defined ad Medium")
    pass

# Replace the LightningModule with a plain PyTorch training loop using the same MoCo components.

# Build models
encoder = build_encoder_from_config(config.model)
backbone = nn.Sequential(encoder).to(device)
projection_head = MoCoProjectionHead(128, 256, 128).to(device)

backbone_momentum = copy.deepcopy(backbone).to(device)
projection_head_momentum = copy.deepcopy(projection_head).to(device)
deactivate_requires_grad(backbone_momentum)
deactivate_requires_grad(projection_head_momentum)

criterion = NTXentLoss(temperature=0.1, memory_bank_size=(memory_bank_size, 128))

# Optimizer & scheduler (only for online encoder & projection head)
optimizer = torch.optim.SGD(
    list(backbone.parameters()) + list(projection_head.parameters()),
    lr=6e-2,
    momentum=0.9,
    weight_decay=5e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# Training loop
backbone.train()
projection_head.train()
backbone_momentum.eval()
projection_head_momentum.eval()

# print dataset sizes (samples & batches) once before training
try:
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    print(f"Train samples: {train_samples}  |  Val samples: {val_samples}")
except Exception:
    print("Could not determine dataset sample counts.")
print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

for epoch in range(max_epochs):
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]", total=len(train_loader))
    for batch in pbar:
        x_q, x_k, _, _ = batch
        x_q = x_q.to(device)
        x_k = x_k.to(device)

        # update momentum encoders (m remains constant per-step here)
        update_momentum(backbone, backbone_momentum, 0.99)
        update_momentum(projection_head, projection_head_momentum, 0.99)

        # forward online (query) encoder
        q = backbone(x_q).flatten(start_dim=1)
        q = projection_head(q)

        # forward momentum (key) encoder with batch shuffle/unshuffle
        k_shuffled, shuffle = batch_shuffle(x_k)
        k_shuffled = k_shuffled.to(device)
        k = backbone_momentum(k_shuffled).flatten(start_dim=1)
        k = projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = criterion(q, k)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(avg_loss=f"{running_loss / max(1, num_batches):.6f}")

    scheduler.step()

    avg_loss = running_loss / max(1, num_batches)
    print(f"Epoch [{epoch+1}/{max_epochs}]  avg_train_loss_ssl: {avg_loss:.6f}")

# Save final online encoder + projection head
out_dir = Path("saved_models")
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "backbone_state_dict": backbone.state_dict(),
        "projection_head_state_dict": projection_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": str(config_file),
    },
    out_dir / "moco_model_final.pt",
)