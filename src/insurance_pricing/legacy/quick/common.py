from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insurance_pricing._typing import FloatArray, as_float_array


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def safe_read_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()


def safe_read_json(path: str | Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {} if default is None else dict(default)
    try:
        return dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return {} if default is None else dict(default)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    y = as_float_array(y_true)
    p = as_float_array(y_pred)
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return float("nan")
    d = y[mask] - p[mask]
    return float(np.sqrt(np.mean(np.square(d))))


__all__ = [
    "safe_read_csv",
    "safe_read_parquet",
    "safe_read_json",
    "safe_float",
    "rmse",
]
