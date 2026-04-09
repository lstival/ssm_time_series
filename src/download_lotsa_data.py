"""Download all Salesforce/lotsa_data subsets to the lustre no-backup cache.

Usage:
    python src/download_lotsa_data.py --cache_dir /lustre/nobackup/WUR/stiva001/hf_datasets
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "Salesforce/lotsa_data"

# Known subsets as of the MOIRAI paper; used as fallback if the API is unavailable.
_KNOWN_SUBSETS = [
    "airpassengers",
    "australian_electricity_demand",
    "bitbrains_fastdb",
    "bitbrains_rnd",
    "brazil_covid",
    "cif_2016_6",
    "cif_2016_12",
    "cloudops_incidents_rolling_window",
    "cloudops_pods_rolling_window",
    "cmip6_1850_2015_5_625deg",
    "cop_load_4_weeks",
    "cop_pv_4_weeks",
    "covid_deaths",
    "dominick_point",
    "ercot_load",
    "ett_h1",
    "ett_h2",
    "ett_m1",
    "ett_m2",
    "exchange_rate",
    "fgw_series_a",
    "fgw_series_b",
    "fgw_series_c",
    "godaddy",
    "hospital",
    "ili",
    "kdd_cup_2018_without_missing",
    "labor",
    "loop_seattle",
    "m1_monthly",
    "m1_quarterly",
    "m1_yearly",
    "m3_monthly",
    "m3_quarterly",
    "m3_yearly",
    "m3_other",
    "m4_daily",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_weekly",
    "m4_yearly",
    "m5",
    "metr_la",
    "monash_australian_electricity",
    "monash_electricity_weekly",
    "nn5_daily_without_missing",
    "nn5_weekly",
    "oikolab_weather",
    "pedestrian_counts",
    "pems_bay",
    "pems_d04",
    "pems_d07",
    "pems_d08",
    "pv_italy",
    "rideshare_without_missing",
    "saugeenday",
    "solar_10min",
    "solar_4_seconds",
    "solar_weekly",
    "subtab_electricity_hourly",
    "subtab_electricity_weekly",
    "sunspot_without_missing",
    "taxi_30min",
    "temperature_rain_without_missing",
    "traffic_hourly",
    "traffic_weekly",
    "us_births",
    "wind_4_seconds",
    "wind_farms_minutely",
]


def _get_all_configs(repo_id: str) -> list[str]:
    """Fetch all config names from HuggingFace, fall back to known list on error."""
    import datasets

    try:
        configs = datasets.get_dataset_config_names(repo_id)
        print(f"Found {len(configs)} subsets via HuggingFace API.")
        return list(configs)
    except Exception as exc:
        print(f"Could not fetch config names from API ({exc}), using known subset list.")
        return list(_KNOWN_SUBSETS)


def download_all(cache_dir: str, skip_existing: bool = True) -> None:
    import datasets

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    print(f"HF_DATASETS_CACHE -> {cache_dir}")

    configs = _get_all_configs(REPO_ID)
    total = len(configs)
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []

    for idx, config in enumerate(configs, start=1):
        print(f"\n[{idx}/{total}] {config}", flush=True)

        # Quick existence check: datasets stores files under a deterministic path.
        expected = Path(cache_dir) / "downloads" / "extracted"
        if skip_existing and expected.exists():
            # A rough heuristic — just attempt a load from cache.
            pass

        try:
            ds = datasets.load_dataset(
                REPO_ID,
                config,
                split="train",
                download_mode="reuse_cache_if_exists",
            )
            rows = len(ds)
            cols = ds.column_names
            print(f"  OK — {rows:,} rows, columns: {cols}")
        except Exception as exc:
            err = str(exc)
            print(f"  FAILED — {err}")
            failed.append((config, err))

    print("\n" + "=" * 60)
    print(f"Download complete. {total - len(failed)} / {total} subsets OK.")
    if failed:
        print(f"\nFailed subsets ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err}")
    print(f"\nData stored in: {cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Salesforce/lotsa_data to lustre cache")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/lustre/nobackup/WUR/AIN/stiva001/hf_datasets",
        help="Destination directory on the lustre no-backup filesystem",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-download even if a subset is already cached",
    )
    args = parser.parse_args()

    download_all(cache_dir=args.cache_dir, skip_existing=not args.no_skip)


if __name__ == "__main__":
    main()
