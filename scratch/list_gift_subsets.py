import datasets
repo_id = "Salesforce/GiftEval"
try:
    subsets = datasets.get_dataset_config_names(repo_id)
    print(f"Subsets for {repo_id}:")
    print(subsets)
except Exception as e:
    print(f"Error listing subsets: {e}")

repo_id_parquet = "Salesforce/GiftEvalParquet"
try:
    subsets = datasets.get_dataset_config_names(repo_id_parquet)
    print(f"\nSubsets for {repo_id_parquet}:")
    print(subsets)
except Exception as e:
    print(f"Error listing subsets for parquet: {e}")
