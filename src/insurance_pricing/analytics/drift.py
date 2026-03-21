from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from insurance_pricing.data.schema import ID_COLS, INDEX_COL, TARGET_FREQ_COL, TARGET_SEV_COL

from .quality import _safe_series


def _psi_from_series(train_s: pd.Series, test_s: pd.Series, bins: int = 10) -> float:
    a = pd.to_numeric(train_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(test_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(a, qs))
    if len(cuts) < 3:
        return float("nan")
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    a_bin = pd.cut(a, bins=cuts, include_lowest=True)
    b_bin = pd.cut(b, bins=cuts, include_lowest=True)
    a_dist = a_bin.value_counts(normalize=True).sort_index()
    b_dist = b_bin.value_counts(normalize=True).sort_index()
    idx = a_dist.index.union(b_dist.index)
    a_p = a_dist.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    b_p = b_dist.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    eps = 1e-6
    a_p = np.clip(a_p, eps, None)
    b_p = np.clip(b_p, eps, None)
    return float(np.sum((a_p - b_p) * np.log(a_p / b_p)))


def compute_drift_numeric_ks_psi(
    train: pd.DataFrame,
    test: pd.DataFrame,
    num_cols: list[str],
    bins: int = 10,
    *,
    include_ids: bool = False,
    exclude_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    exclude = set(str(c) for c in (exclude_cols or []))
    if not include_ids:
        exclude.update([INDEX_COL, *ID_COLS, TARGET_FREQ_COL, TARGET_SEV_COL])
    rows = []
    for c in num_cols:
        if c in exclude:
            continue
        if c not in train.columns or c not in test.columns:
            continue
        tr = _safe_series(train[c]).replace([np.inf, -np.inf], np.nan).dropna()
        te = _safe_series(test[c]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(tr) < 5 or len(te) < 5:
            continue
        ks = stats.ks_2samp(tr.to_numpy(dtype=float), te.to_numpy(dtype=float), method="auto")
        rows.append(
            {
                "column": c,
                "n_train_non_null": int(len(tr)),
                "n_test_non_null": int(len(te)),
                "mean_train": float(np.mean(tr)),
                "mean_test": float(np.mean(te)),
                "median_train": float(np.median(tr)),
                "median_test": float(np.median(te)),
                "std_train": float(np.std(tr)),
                "std_test": float(np.std(te)),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "psi": _psi_from_series(tr, te, bins=bins),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "n_train_non_null",
                "n_test_non_null",
                "mean_train",
                "mean_test",
                "median_train",
                "median_test",
                "std_train",
                "std_test",
                "ks_stat",
                "ks_pvalue",
                "psi",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["psi", "ks_stat"], ascending=[False, False])
        .reset_index(drop=True)
    )


def compute_drift_categorical_chi2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
    top_k: int = 50,
) -> pd.DataFrame:
    rows = []
    for c in cat_cols:
        if c not in train.columns or c not in test.columns:
            continue
        tr = train[c].astype(str).fillna("NA")
        te = test[c].astype(str).fillna("NA")
        tr_vc = tr.value_counts(dropna=False)
        te_vc = te.value_counts(dropna=False)
        seen_tr = set(tr_vc.index.astype(str))
        seen_te = set(te_vc.index.astype(str))
        unseen = seen_te - seen_tr
        top_levels = tr_vc.head(top_k).index.astype(str).tolist()
        levels = sorted(set(top_levels) | set(te_vc.head(top_k).index.astype(str).tolist()))
        if len(levels) < 2:
            chi2_stat, pvalue, dof = np.nan, np.nan, np.nan
        else:
            cont = np.vstack(
                [
                    tr_vc.reindex(levels, fill_value=0).to_numpy(dtype=float),
                    te_vc.reindex(levels, fill_value=0).to_numpy(dtype=float),
                ]
            )
            try:
                chi2_stat, pvalue, dof, _ = stats.chi2_contingency(cont)
            except Exception:
                chi2_stat, pvalue, dof = np.nan, np.nan, np.nan
        rows.append(
            {
                "column": c,
                "train_unique": int(tr.nunique(dropna=False)),
                "test_unique": int(te.nunique(dropna=False)),
                "unseen_test_levels": int(len(unseen)),
                "unseen_ratio_on_test_levels": float(len(unseen) / max(len(seen_te), 1)),
                "unseen_ratio_on_test_rows": float(te.isin(list(unseen)).mean()) if unseen else 0.0,
                "chi2_stat_topk": float(chi2_stat) if np.isfinite(chi2_stat) else np.nan,
                "chi2_pvalue_topk": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "chi2_dof_topk": float(dof) if np.isfinite(dof) else np.nan,
                "top_train_levels": " | ".join(map(str, top_levels[:10])),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["unseen_ratio_on_test_rows", "unseen_test_levels"], ascending=[False, False])
        .reset_index(drop=True)
    )
