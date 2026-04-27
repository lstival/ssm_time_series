import datasets
repo_id = "Salesforce/GiftEvalParquet"
subset = "ett1_H_short"
try:
    ds = datasets.load_dataset(repo_id, subset, split="train")
    print(f"Features for {subset}:")
    print(ds.features)
    print("\nFirst sample keys:")
    print(ds[0].keys())
    # print("\nFirst sample target length:")
    # print(len(ds[0]['target']))
except Exception as e:
    print(f"Error loading {subset}: {e}")
