import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, Sequence, Value
from typing import Dict, List, Optional, Sequence, Tuple


def _cast_target_dtype(ds: datasets.Dataset, dtype: str) -> datasets.Dataset:
    if "target" not in ds.column_names:
        return ds

    target_feature = ds.features["target"]

    def _is_current_dtype(feature) -> bool:
        if isinstance(feature, datasets.Sequence):
            inner_feature = feature.feature
            return isinstance(inner_feature, datasets.Value) and inner_feature.dtype == dtype
        if isinstance(feature, datasets.Value):
            return feature.dtype == dtype
        return False

    if _is_current_dtype(target_feature):
        return ds

    def _build_feature(feature):
        if isinstance(feature, datasets.Sequence):
            length = feature.length
            return datasets.Sequence(feature=datasets.Value(dtype), length=length)
        if isinstance(feature, datasets.Value):
            return datasets.Value(dtype)
        return feature

    new_feature = _build_feature(target_feature)
    try:
        return ds.cast_column("target", new_feature)
    except (TypeError, ValueError):
        return ds


def load_chronos_datasets(
    dataset_names: Sequence[str],
    split: str = "train",
    *,
    repo_id: str = "autogluon/chronos_datasets",
    set_numpy_format: bool = True,
    target_dtype: Optional[str] = "float64",
    **load_kwargs,
) -> datasets.Dataset:
    """Load and concatenate Chronos datasets from Hugging Face."""
    if not dataset_names:
        raise ValueError("dataset_names must be a non-empty sequence.")

    loaded_datasets: List[Tuple[str, datasets.Dataset]] = []
    for name in dataset_names:
        ds = datasets.load_dataset(repo_id, name, split=split, **load_kwargs)
        if "target" not in ds.column_names:
            # Rename common consumption columns to target for downstream compatibility
            for candidate in ("consumption_kW", "power_mw"):
                if candidate in ds.column_names:
                    ds = ds.rename_column(candidate, "target")
                    break

        # Convert the data for float64 (allow concat)
        if target_dtype is not None:
            ds = _cast_target_dtype(ds, target_dtype)

        # Check if the dataset have the "target" colummn
        if "target" not in ds.column_names:
            print(f"Skipping dataset '{name}': missing 'target' column.")
            continue

        if len(ds.column_names) > 1:
            ds = ds.select_columns(["target"])

        loaded_datasets.append((name, ds))

    combined, _ = _concatenate_with_reporting(loaded_datasets)

    if set_numpy_format:
        combined.set_format("numpy")  # sequences returned as numpy arrays

    return combined


def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
    """Convert dataset to long data frame format."""
    sequence_columns = [col for col in ds.features if isinstance(ds.features[col], datasets.Sequence)]
    return ds.to_pandas().explode(sequence_columns).infer_objects()


def _infer_numpy_dtype(feature) -> Optional[np.dtype]:
    """Derive the closest numpy dtype supported by the provided feature."""
    if isinstance(feature, datasets.Sequence):
        return _infer_numpy_dtype(feature.feature)
    if isinstance(feature, datasets.Value):
        try:
            return np.dtype(feature.dtype)
        except TypeError:
            return None
    return None


def _concatenate_with_reporting(
    datasets_with_names: Sequence[Tuple[str, datasets.Dataset]]
) -> Tuple[datasets.Dataset, Dict[str, str]]:
    """Concatenate datasets sequentially while logging failures."""
    kept: List[datasets.Dataset] = []
    failures: Dict[str, str] = {}

    reference: Optional[datasets.Dataset] = None

    for name, ds in datasets_with_names:
        if reference is None:
            kept.append(ds)
            reference = ds
            continue

        try:
            datasets.concatenate_datasets([reference, ds])
        except (TypeError, ValueError) as exc:
            error_message = str(exc)
            failures[name] = error_message
            print(f"Skipping dataset '{name}' during concatenation: {error_message}")
            continue

        kept.append(ds)

    if not kept:
        raise ValueError("No datasets remained for concatenation after filtering failures.")

    combined = kept[0] if len(kept) == 1 else datasets.concatenate_datasets(kept)

    if failures:
        failed_names = ", ".join(failures.keys())
        print(f"Datasets skipped due to concatenation errors: {failed_names}")

    return combined, failures


def target_only_view(
    ds: datasets.Dataset,
    *,
    keep_format: bool = True,
) -> datasets.Dataset:
    """Return a dataset view that exposes only the ``target`` column."""
    if "target" not in ds.column_names:
        raise KeyError("Dataset does not contain a 'target' column.")

    target_dataset = ds.select_columns(["target"]) if len(ds.column_names) > 1 else ds

    if not keep_format:
        target_dataset.reset_format()
        return target_dataset

    format_info = getattr(ds, "format", {}) or {}
    format_type = format_info.get("type")
    format_kwargs = dict(format_info.get("format_kwargs", {})) if format_info else {}

    if format_type == "numpy":
        inferred_dtype = _infer_numpy_dtype(target_dataset.features["target"])
        if inferred_dtype is not None:
            format_kwargs["dtype"] = inferred_dtype
        else:
            format_kwargs.pop("dtype", None)

    if format_type:
        try:
            return target_dataset.with_format(
                type=format_type,
                columns=["target"],
                output_all_columns=False,
                format_kwargs=format_kwargs,
            )
        except ValueError:
            target_dataset.reset_format()
            return target_dataset

    target_dataset.reset_format()
    return target_dataset


def filter_target_dtype(
    datasets_by_name: Dict[str, Dataset],
    expected_dtype: str = "float64",
) -> Tuple[Dict[str, Dataset], Dict[str, str]]:
    kept: Dict[str, Dataset] = {}
    skipped: Dict[str, str] = {}
    for name, ds in datasets_by_name.items():
        try:
            target_feature = ds.features["target"]
        except KeyError:
            skipped[name] = "missing 'target' feature"
            print(f"Skipping dataset '{name}': {skipped[name]}")
            continue

        if isinstance(target_feature, Sequence):
            actual_dtype = target_feature.feature.dtype
        elif isinstance(target_feature, Value):
            actual_dtype = target_feature.dtype
        else:
            actual_dtype = getattr(target_feature, "dtype", None)

        if actual_dtype != expected_dtype:
            skipped[name] = f"target dtype {actual_dtype!r} != {expected_dtype!r}"
            print(f"Skipping dataset '{name}': {skipped[name]}")
            continue

        kept[name] = ds
    return kept, skipped


if __name__ == "__main__":
    ## https://huggingface.co/datasets/autogluon/chronos_datasets/tree/main
    ## "weatherbench_weekly"
    datasets_to_load = [
        "m4_daily",
        "m4_hourly",
        "m4_monthly",
        "m4_yearly",
        "monash_australian_electricity",
        "taxi_30min",
        "monash_traffic",
        "monash_kdd_cup_2018",
        "m5",
        "mexico_city_bikes",
        "exchange_rate",
        "monash_car_parts",
        "monash_covid_deaths",
        "monash_electricity_hourly",
        "monash_fred_md",
        "monash_hospital",
        "monash_m1_monthly",
        "monash_m1_quarterly",
        "monash_m1_yearly",
        "monash_m3_monthly",
        "monash_m3_quarterly",
        "monash_m3_yearly",
        "monash_nn5_weekly",
        "taxi_30min",
        "uber_tlc_daily",
        "uber_tlc_hourly",
        "wind_farms_hourly",
        "wind_farms_daily",
        "dominick",
        "electricity_15min",
        "solar_1h",
        # "ercot", # 8 rows with 158k length time series
    ]

    dataset = load_chronos_datasets(datasets_to_load, target_dtype="float64")
    print(to_pandas(dataset).head())
    print(dataset)
