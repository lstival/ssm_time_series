import torch
from cm_mamba.models.mamba_encoder import MambaEncoder
from cm_mamba.models.mamba_visual_encoder import MambaVisualEncoder
from cm_mamba.models.dual_forecast import DualEncoderForecastRegressor, DualEncoderForecastMLP

def run_inference_demo():
    # 1. Setup Temporal Encoder
    temporal_encoder = MambaEncoder(
        input_dim=32,
        model_dim=128,
        depth=4,
        embedding_dim=64
    )
    
    # 2. Setup Visual Encoder
    visual_encoder = MambaVisualEncoder(
        input_dim=32,
        model_dim=128,
        depth=4,
        embedding_dim=64
    )
    
    # 3. Setup Dual Forecast Model
    head = DualEncoderForecastMLP(
        input_dim=128, # sum of embedding dims
        hidden_dim=256,
        horizons=[96, 720]
    )
    
    model = DualEncoderForecastRegressor(
        encoder=temporal_encoder,
        visual_encoder=visual_encoder,
        head=head
    )
    
    # 4. Dummy Input (Batch, Time, Features)
    x = torch.randn(1, 96, 32)
    
    # 5. Forward Pass
    with torch.no_grad():
        # Get embeddings
        temp_emb = temporal_encoder(x)
        vis_emb = visual_encoder(x)
        print(f"Temporal Embedding: {temp_emb.shape}")
        print(f"Visual Embedding: {vis_emb.shape}")
        
        # Get forecast
        forecast = model(x, horizon=96)
        print(f"Forecast (96 steps): {forecast.shape}")

if __name__ == "__main__":
    run_inference_demo()
