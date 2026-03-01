from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from .schema import INDEX_COL, TARGET_FREQ_COL, TARGET_SEV_COL

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_train_test(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    return train, test

def build_targets(train: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    y_sev = train[TARGET_SEV_COL].astype(float).rename("y_sev")
    y_freq = (y_sev > 0).astype(int).rename("y_freq")
    return y_freq, y_sev


def validate_data_contract(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    required_train = {TARGET_FREQ_COL, TARGET_SEV_COL, INDEX_COL}
    required_test = {INDEX_COL}
    missing_train = sorted(required_train - set(train.columns))
    missing_test = sorted(required_test - set(test.columns))
    common_cols = sorted(set(train.columns).intersection(set(test.columns)))
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "missing_train_columns": missing_train,
        "missing_test_columns": missing_test,
        "n_common_columns": int(len(common_cols)),
    }

