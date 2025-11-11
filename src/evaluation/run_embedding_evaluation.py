"""Run embedding cache evaluation with minimal configuration."""

from pathlib import Path
import sys

# Add project paths
SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
for path in (SRC_DIR, ROOT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.down_tasks.evaluate_icml import EvalConfig


def main():
    """Run evaluation with test split only."""
    
    # Configure for test split evaluation only
    config = EvalConfig(
        checkpoint_path=r"C:\WUR\ssm_time_series\checkpoints\multi_horizon_forecast_emb_128_tgt_1_20251108_1124\best_model.pt",
        embedding_cache_dir=r"C:\WUR\ssm_time_series\embedding_cache", 
        split="test",  # Only evaluate on test split
        batch_size=64,
        num_workers=4,
        dataset_filter=["electricity", "weather"],  # Limit to specific datasets for speed
        generate_plots=True,
        samples_per_plot=2,
        show_plots=False  # Save plots but don't show them
    )
    
    print("Starting embedding cache evaluation...")
    print(f"Configuration:")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print(f"  Embedding cache: {config.embedding_cache_dir}")
    print(f"  Split: {config.split}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Dataset filter: {config.dataset_filter}")
    print(f"  Generate plots: {config.generate_plots}")
    
    # Import evaluation functions
    from src.down_tasks.evaluate_icml import (
        discover_embedding_datasets, 
        evaluate_all_embedding_datasets,
        evaluate_embedding_dataset
    )
    from src.down_tasks.forecast_utils import (
        load_trained_model,
        save_evaluation_results,
        print_evaluation_summary,
        plot_forecast_comparison
    )
    from util import default_device
    
    try:
        # Setup paths
        checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        embedding_cache_dir = Path(config.embedding_cache_dir).expanduser().resolve()
        results_dir = Path(config.results_dir).expanduser().resolve()
        plots_dir = Path(config.plots_dir).expanduser().resolve()

        # Validate paths
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not embedding_cache_dir.exists():
            raise FileNotFoundError(f"Embedding cache directory not found: {embedding_cache_dir}")

        results_dir.mkdir(parents=True, exist_ok=True)
        if config.generate_plots:
            plots_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        device = default_device()
        print(f"Using device: {device}")

        # Load trained model
        print(f"Loading model from: {checkpoint_path}")
        model, checkpoint_info = load_trained_model(
            checkpoint_path, device, config.mlp_hidden_dim
        )

        # Get horizons to evaluate
        if config.horizons:
            horizons = [int(h.strip()) for h in config.horizons.split(",")]
            print(f"Using custom horizons: {horizons}")
        else:
            horizons = checkpoint_info.get('horizons')
            print(f"Using model's training horizons: {horizons}")

        # Evaluate on all embedding cache datasets
        print(f"\nEvaluating on embedding cache datasets ({config.split} split)...")
        all_results = evaluate_all_embedding_datasets(
            model=model,
            embedding_cache_dir=embedding_cache_dir,
            horizons=horizons,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            split=config.split
        )

        # Filter results if dataset filter is provided
        if config.dataset_filter:
            filtered_results = {
                k: v for k, v in all_results.items()
                if any(filter_name.lower() in k.lower() for filter_name in config.dataset_filter)
            }
            if not filtered_results:
                print(f"Warning: No datasets matched filter {config.dataset_filter}")
                filtered_results = all_results
            all_results = filtered_results
            print(f"Filtered to {len(all_results)} datasets: {list(all_results.keys())}")

        # Print summary
        print_evaluation_summary(all_results, checkpoint_info)

        # Save results
        json_path, csv_path = save_evaluation_results(
            all_results, checkpoint_info, results_dir, f"embedding_cache_evaluation_{config.split}"
        )

        print(f"\nResults saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")

        # Generate plots if requested
        if config.generate_plots:
            print(f"\nGenerating forecast plots...")

            # Get predictions for plotting (evaluate with return_predictions=True)
            plot_results = {}
            datasets = discover_embedding_datasets(embedding_cache_dir)
            
            # Get the first 2 datasets for plotting
            datasets_to_plot = datasets[:2]
            
            for group, dataset_name, dataset_path in datasets_to_plot:
                dataset_key = f"{group}/{dataset_name}"
                if config.dataset_filter:
                    if not any(filter_name.lower() in dataset_key.lower() for filter_name in config.dataset_filter):
                        continue
                        
                print(f"Getting predictions for {dataset_key}...")

                dataset_predictions = evaluate_embedding_dataset(
                    model=model,
                    dataset_path=dataset_path,
                    horizons=horizons,
                    device=device,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split=config.split,
                    return_predictions=True
                )
                plot_results[dataset_key] = dataset_predictions

            # Generate plots
            figures = plot_forecast_comparison(
                results_dict=plot_results,
                datasets_to_plot=None,  # Plot all available
                horizons_to_plot=horizons[:2] if len(horizons) > 2 else horizons,  # Limit horizons for plotting
                samples_per_dataset=config.samples_per_plot,
                features_to_plot=[0],  # Plot first feature only
                save_dir=plots_dir,
                show_plots=config.show_plots
            )

            print(f"Generated {len(figures)} forecast plots in: {plots_dir}")

        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()