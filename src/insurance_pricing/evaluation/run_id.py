from __future__ import annotations

import pandas as pd


def make_run_id_from_df(df: pd.DataFrame) -> pd.Series:
    if "run_id" in df.columns:
        return df["run_id"].astype(str)

    cols_new = [
        "feature_set",
        "engine",
        "family",
        "tweedie_power",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
    ]
    if all(c in df.columns for c in cols_new):
        out = df[cols_new[0]].astype(str)
        for c in cols_new[1:]:
            out = out + "|" + df[c].astype(str)
        return out

    cols_legacy_v2 = [
        "feature_set",
        "engine",
        "family",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
    ]
    if all(c in df.columns for c in cols_legacy_v2):
        out = df[cols_legacy_v2[0]].astype(str)
        for c in cols_legacy_v2[1:]:
            out = out + "|" + df[c].astype(str)
        return out

    cols_v1 = ["engine", "config_id", "seed", "severity_mode", "calibration"]
    if all(c in df.columns for c in cols_v1):
        out = df[cols_v1[0]].astype(str)
        for c in cols_v1[1:]:
            out = out + "|" + df[c].astype(str)
        return out

    raise KeyError("Cannot build run_id: required columns are missing.")
