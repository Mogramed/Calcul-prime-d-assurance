from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x.astype(str), y.astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    try:
        chi2, _, _, _ = stats.chi2_contingency(tab)
    except Exception:
        return float("nan")
    n = tab.values.sum()
    if n <= 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2corr / denom))

def compute_cramers_v_table(df: pd.DataFrame, cat_cols: list[str], max_cols: int = 12) -> pd.DataFrame:
    cols = [c for c in cat_cols if c in df.columns][:max_cols]
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            val = _cramers_v(df[c1], df[c2])
            rows.append({"col_a": c1, "col_b": c2, "cramers_v": val})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)

