from __future__ import annotations

import numpy as np
import pandas as pd

from src.insurance_pricing.data.schema import INDEX_COL

def build_submission(index_series: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    sub = pd.DataFrame({INDEX_COL: index_series.astype(int).to_numpy(), "pred": np.asarray(pred, dtype=float)})
    sub["pred"] = sub["pred"].clip(lower=0.0)
    return sub

