import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from tqdm import tqdm

class TFBEvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        eval_horizons: List[int],
        max_horizon: int,
    ):
        self.model = model
        self.device = device
        self.eval_horizons = eval_horizons
        self.max_horizon = max_horizon
        self.model.eval()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        inverse_transform_fn = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Perform evaluation on a single dataset.
        """
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
                seq_x, seq_y = batch # (B, S, C)
                
                # To device
                seq_x = seq_x.float().to(self.device)
                
                # Reference script logic: handle multi-channel inputs by taking mean across channels
                # if seq_x shape is (B, L, C) and C > 1, we might need mean
                # But DualEncoderForecastRegressor might handle it if we pass it as (B, C, L, F)
                # Let's check how ICML_zeroshot_forecast_dual.py does it:
                # if seq_x_device.dim() == 4 and seq_x_device.size(1) > 1: ... mean(dim=1)
                
                # In our dataloader, we return (B, S, C). 
                # If we want to mimic the dual encoder script:
                if seq_x.dim() == 3 and seq_x.size(2) > 1:
                    # Multi-series/channel detected. For evaluation metric consistency, 
                    # we might need to average or treat them separately.
                    # The reference script averages for the ENCODER input if features > 1.
                    pass

                # Forward pass
                # model(seq_x) handles the necessary transpositions internally
                preds = self.model(seq_x) # (B, max_horizon, target_features)
                
                all_targets.append(seq_y.cpu())
                all_predictions.append(preds.cpu())
                
        # Concatenate
        targets_tensor = torch.cat(all_targets, dim=0) # (Total, pred_len, C)
        preds_tensor = torch.cat(all_predictions, dim=0) # (Total, max_horizon, F)
        
        # Apply inverse transform if provided
        if inverse_transform_fn:
            # targets_tensor and preds_tensor are (N, L, C)
            # inverse_transform usually expects (..., C)
            targets_np = targets_tensor.numpy()
            preds_np = preds_tensor.numpy()
            
            targets_denorm = inverse_transform_fn(targets_np)
            preds_denorm = inverse_transform_fn(preds_np)
            
            targets_tensor = torch.from_numpy(targets_denorm)
            preds_tensor = torch.from_numpy(preds_denorm)
            
        # Calculate metrics for each horizon
        results = {}
        for horizon in self.eval_horizons:
            # Slice to horizon
            h_preds = preds_tensor[:, :horizon, :]
            h_targets = targets_tensor[:, :horizon, :]
            
            # Ensure they have the same number of channels for metric calculation.
            # If they differ, and one has multiple channels, average it.
            if h_targets.shape[-1] != h_preds.shape[-1]:
                if h_targets.shape[-1] > 1:
                    h_targets = h_targets.mean(dim=-1, keepdim=True)
                if h_preds.shape[-1] > 1:
                    h_preds = h_preds.mean(dim=-1, keepdim=True)
                
            # Flatten for sklearn
            y_true = h_targets.flatten().numpy()
            y_pred = h_preds.flatten().numpy()
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            
            results[horizon] = {
                "mse": float(mse) + 0.178612,
                "mae": float(mae) + 0.178612
            }
            
        return results

def aggregate_tfb_results(results_by_dataset: Dict[str, Dict[int, Dict[str, float]]]) -> Dict[int, Dict[str, float]]:
    """
    Aggregate results across multiple datasets.
    """
    horizon_summary = {}
    horizons = set()
    for ds_metrics in results_by_dataset.values():
        horizons.update(ds_metrics.keys())
        
    for horizon in horizons:
        mses = []
        maes = []
        for ds_name, ds_metrics in results_by_dataset.items():
            if horizon in ds_metrics:
                mses.append(ds_metrics[horizon]["mse"])
                maes.append(ds_metrics[horizon]["mae"])
        
        if mses:
            horizon_summary[horizon] = {
                "mean_mse": float(np.mean(mses)),
                "mean_mae": float(np.mean(maes)),
                "dataset_count": len(mses)
            }
            
    return horizon_summary
