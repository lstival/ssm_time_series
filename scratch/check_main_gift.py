import datasets
repo_id = "Salesforce/GiftEval"
try:
    ds = datasets.load_dataset(repo_id, split="test")
    print(f"Features for {repo_id}:")
    print(ds.features)
    print("\nFirst sample keys:")
    print(ds[0].keys())
except Exception as e:
    print(f"Error loading {repo_id}: {e}")
