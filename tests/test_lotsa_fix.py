import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataloaders.lotsa_dataset import load_lotsa_datasets, LOTSA_DEFAULT_SUBSETS

def test_load():
    # Test a few that were failing
    to_test = [
        "monash_m3_monthly",    # Renamed from m3_monthly
        "exchange_rate",        # Exists in autogluon/chronos_datasets
        "ett_h1",               # Exists in autogluon/chronos_datasets
        "monash_m3_yearly",     # Renamed from m3_yearly
    ]
    
    print(f"Attempting to load subsets: {to_test}")
    try:
        # We allow online access for this test to verify the logic works if they aren't in cache
        # But for the user, they'll likely be in the home cache or they can download them.
        ds = load_lotsa_datasets(to_test, force_offline=False)
        print(f"\nSuccessfully loaded {len(to_test)} datasets!")
        print(f"Total rows: {len(ds):,}")
        print(f"Datasets: {to_test}")
    except Exception as e:
        print(f"\nLoad failed: {e}")

if __name__ == "__main__":
    test_load()
