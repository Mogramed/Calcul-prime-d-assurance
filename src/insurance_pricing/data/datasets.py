from __future__ import annotations

from pathlib import Path

import pandas as pd

from insurance_pricing.data.io import load_train_test, validate_data_contract
from insurance_pricing.data.schema import DatasetBundle
from insurance_pricing.features.engineering import prepare_feature_sets


def load_datasets(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_train_test(data_dir)


def build_feature_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    drop_identifiers: bool = True,
) -> dict[str, DatasetBundle]:
    return prepare_feature_sets(
        train,
        test,
        rare_min_count=30,
        drop_identifiers=drop_identifiers,
    )


def select_bundle(
    feature_sets: dict[str, DatasetBundle],
    feature_set_name: str,
) -> DatasetBundle:
    if feature_set_name not in feature_sets:
        available = ", ".join(sorted(feature_sets.keys()))
        raise KeyError(f"Unknown feature_set '{feature_set_name}'. Available: {available}")
    return feature_sets[feature_set_name]


__all__ = [
    "load_datasets",
    "build_feature_sets",
    "select_bundle",
    "validate_data_contract",
]
