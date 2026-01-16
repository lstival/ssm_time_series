import torch
import sys
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation_tfb.zoo_models import get_model

def test_model(model_name: str):
    print(f"\nTesting {model_name}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(model_name, device=device)
        
        # Create dummy context (Batch=1, Seq=96, Features=1)
        context = torch.randn(1, 96, 1).to(device)
        horizon = 96
        
        with torch.no_grad():
            preds = model.predict(context, horizon)
            
        print(f"SUCCESS: {model_name} output shape: {preds.shape}")
        return True
    except Exception as e:
        print(f"FAILURE: {model_name} failed with error: {e}")
        return False

if __name__ == "__main__":
    models_to_test = ["chronos", "transformer", "patchtst"] 
    # Skipping timesfm by default as it might require heavy weights or complex setup
    # but the user asked for "all models"
    models_to_test.append("timesfm")
    
    results = {}
    for model in models_to_test:
        results[model] = test_model(model)
        
    print("\n" + "="*30)
    print("SUMMARY")
    print("="*30)
    for model, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{model:15}: {status}")
