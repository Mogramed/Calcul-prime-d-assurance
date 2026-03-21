from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _smooth_target_encoding_map(
    x: pd.Series,
    y: pd.Series,
    *,
    smoothing: float,
) -> tuple[dict[str, float], float]:
    xx = x.astype(str)
    yy = y.astype(float)
    prior = float(np.nanmean(yy))
    grp = pd.DataFrame({"x": xx, "y": yy}).groupby("x")["y"].agg(["sum", "count"])
    val = (grp["sum"] + prior * smoothing) / (grp["count"] + smoothing)
    return {str(k): float(v) for k, v in val.items()}, prior


def _add_fold_target_encoding(
    *,
    X_tr: pd.DataFrame,
    y_freq_tr: np.ndarray,
    y_sev_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    cols: Sequence[str],
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xtr = X_tr.copy()
    xva = X_va.copy()
    xte = X_te.copy()
    yy_freq = pd.Series(y_freq_tr)
    yy_sev = pd.Series(y_sev_tr.astype(float))

    for c in cols:
        if c not in xtr.columns:
            continue
        m_freq, prior_freq = _smooth_target_encoding_map(xtr[c], yy_freq, smoothing=smoothing)
        m_sev, prior_sev = _smooth_target_encoding_map(xtr[c], yy_sev, smoothing=smoothing)

        def _map(s: pd.Series, mapping: Mapping[str, float], prior: float) -> pd.Series:
            return s.astype(str).map(mapping).astype(float).fillna(prior)

        xtr[f"te_freq_{c}"] = _map(xtr[c], m_freq, prior_freq)
        xva[f"te_freq_{c}"] = _map(xva[c], m_freq, prior_freq)
        xte[f"te_freq_{c}"] = _map(xte[c], m_freq, prior_freq)

        xtr[f"te_sev_{c}"] = _map(xtr[c], m_sev, prior_sev)
        xva[f"te_sev_{c}"] = _map(xva[c], m_sev, prior_sev)
        xte[f"te_sev_{c}"] = _map(xte[c], m_sev, prior_sev)
    return xtr, xva, xte


def _apply_winsor(y: np.ndarray, quantile: float) -> np.ndarray:
    yy = np.asarray(y, dtype=float)
    q = float(np.nanquantile(yy, quantile))
    return np.minimum(yy, q)


def _smearing_inverse(
    y_pos: np.ndarray,
    z_tr: np.ndarray,
    z_va: np.ndarray,
    z_te: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    y_log = np.log1p(np.asarray(y_pos, dtype=float))
    resid = y_log - np.asarray(z_tr, dtype=float)
    if sample_weight is None:
        smear = float(np.mean(np.exp(resid)))
    else:
        smear = float(np.average(np.exp(resid), weights=np.asarray(sample_weight, dtype=float)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    m_va = np.maximum(smear * np.exp(np.asarray(z_va, dtype=float)) - 1.0, 0.0)
    m_te = np.maximum(smear * np.exp(np.asarray(z_te, dtype=float)) - 1.0, 0.0)
    return m_va, m_te
