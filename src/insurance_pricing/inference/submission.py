from __future__ import annotations

import pandas as pd

from insurance_pricing._typing import FloatArray, as_float_array
from insurance_pricing.data.schema import INDEX_COL


def build_submission(index_series: pd.Series, pred: FloatArray) -> pd.DataFrame:
    sub = pd.DataFrame(
        {INDEX_COL: index_series.astype(int).to_numpy(), "pred": as_float_array(pred)}
    )
    sub["pred"] = sub["pred"].clip(lower=0.0)
    return sub
