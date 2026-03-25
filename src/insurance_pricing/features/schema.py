from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from insurance_pricing.features.engineering import add_engineered_features


def build_feature_frame_for_inference(
    raw_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> pd.DataFrame:
    df = add_engineered_features(raw_df.copy())
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = np.nan
    for col in cat_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).fillna("NA")
    return out


def build_feature_schema(
    feature_cols: Sequence[str], cat_cols: Sequence[str]
) -> dict[str, list[str]]:
    return {
        "feature_cols": list(feature_cols),
        "cat_cols": list(cat_cols),
        "num_cols": [c for c in feature_cols if c not in set(cat_cols)],
    }
