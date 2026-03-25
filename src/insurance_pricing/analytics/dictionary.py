from __future__ import annotations

import numpy as np
import pandas as pd

from .quality import _role_guess


def build_data_dictionary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(train.columns).union(set(test.columns)))
    rows = []
    for c in cols:
        tr_present = c in train.columns
        te_present = c in test.columns
        tr_s = train[c] if tr_present else pd.Series(dtype="object")
        te_s = test[c] if te_present else pd.Series(dtype="object")
        dtype_train = str(tr_s.dtype) if tr_present else None
        dtype_test = str(te_s.dtype) if te_present else None
        rows.append(
            {
                "column": c,
                "present_train": int(tr_present),
                "present_test": int(te_present),
                "dtype_train": dtype_train,
                "dtype_test": dtype_test,
                "nunique_train": int(tr_s.nunique(dropna=False)) if tr_present else np.nan,
                "nunique_test": int(te_s.nunique(dropna=False)) if te_present else np.nan,
                "missing_rate_train": float(tr_s.isna().mean()) if tr_present else np.nan,
                "missing_rate_test": float(te_s.isna().mean()) if te_present else np.nan,
                "sample_values_train": " | ".join(
                    map(str, tr_s.dropna().astype(str).head(3).tolist())
                )
                if tr_present
                else "",
                "role_guess": _role_guess(
                    c, dtype_train or dtype_test or "unknown", tr_present, te_present
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["role_guess", "column"]).reset_index(drop=True)


def classify_columns(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    df = build_data_dictionary(train, test).copy()
    df["is_target"] = df["role_guess"].isin(["target_freq", "target_sev"]).astype(int)
    df["is_id_like"] = df["role_guess"].str.startswith("id_").astype(int)
    df["is_categorical"] = (df["role_guess"] == "categorical").astype(int)
    df["is_numeric"] = (df["role_guess"] == "numeric").astype(int)
    df["high_cardinality_train"] = (
        pd.to_numeric(df["nunique_train"], errors="coerce").fillna(0) >= 100
    ).astype(int)
    return df
