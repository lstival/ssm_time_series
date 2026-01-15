# TFB Forecasting Evaluation Package

This package provides tools to evaluate baseline forecasting datasets in TFB format using a dual-encoder architecture. It supports automated batch evaluation across thousands of datasets, handles short sequences via interpolation, and generates both CSV and LaTeX reports.

## Installation & Requirements

Ensure you are using the `timeseries` conda environment:
```bash
conda activate timeseries
```

## Batch Evaluation

To evaluate all datasets in the TFB directory and generate automated reports, use the `batch_evaluator.py` script.

### Running the Evaluation
Navigate to the project root and run:
```bash
python src/evaluation_tfb/batch_evaluator.py `
  --data_dir "data/forecasting/forecasting" `
  --model_config "src/configs/tfb_eval_model.yaml" `
  --forecast_checkpoint "checkpoints/multi_horizon_forecast_dual_frozen_20251209_1049/all_datasets/best_model.pt" `
  --encoder_checkpoint "checkpoints/ts_encoder_20251126_1750/time_series_best.pt" `
  --visual_encoder_checkpoint "checkpoints/ts_encoder_20251126_1750/visual_encoder_best.pt" `
  --horizons 96 192 336 720 `
  --results_dir "results/tfb_batch_full" `
  --stride 100 `
  --device cpu
```

### Arguments
- `--data_dir`: Directory containing the TFB `.csv` files.
- `--model_config`: Path to the model YAML configuration (use `src/configs/tfb_eval_model.yaml`).
- `--horizons`: List of forecast horizons to evaluate (default: 96, 192, 336, 720).
- `--stride`: Stride for the sliding window. Use a larger value (e.g., 100) to speed up evaluation on large dataset collections.
- `--device`: `cpu` or `cuda`.
- `--max_datasets`: (Optional) Limit evaluation to the first N datasets for testing.

## Saved Results

Results are saved in the directory specified by `--results_dir`:

1.  **`tfb_batch_results.csv`**: A machine-readable file containing MSE and MAE for every dataset and horizon.
2.  **`tfb_batch_results.tex`**: A publication-ready LaTeX table summarizing the results across all datasets.

## Key Features

- **Interpolation**: If a dataset is shorter than the required window (`context_length + prediction_length`), it is automatically linearly interpolated to the minimum required length.
- **Multivariate Support**: Automatically handles channel alignment and averages metrics across channels for consistency with project baselines.
- **Custom Metrics**: Evaluation includes the project-specific offset in MSE and MAE calculations.
