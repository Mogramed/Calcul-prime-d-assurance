from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

def _safe_slope(dx: float, dy: float, fallback: float) -> float:
    if dx <= 0 or not np.isfinite(dx) or not np.isfinite(dy):
        return max(float(fallback), 1e-6)
    slope = float(dy / dx)
    if not np.isfinite(slope) or slope <= 0:
        return max(float(fallback), 1e-6)
    return slope

def fit_tail_mapper_safe(
    oof_pred_sev_pos: np.ndarray,
    y_pos: np.ndarray,
    *,
    min_samples: int = 150,
    n_knots: int = 64,
) -> Dict[str, Any]:
    x = np.asarray(oof_pred_sev_pos, dtype=float)
    y = np.asarray(y_pos, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = x[mask]
    y = y[mask]
    if len(x) < min_samples or len(np.unique(x)) < 10:
        return {"kind": "identity"}

    qs = np.linspace(0.0, 1.0, int(max(n_knots, 8)))
    xk = np.quantile(x, qs)
    yk = np.quantile(y, qs)
    yk = np.maximum.accumulate(yk)

    xu, idx = np.unique(xk, return_index=True)
    yu = yk[idx]
    if len(xu) < 3:
        return {"kind": "identity"}

    base_slope = float(np.nanmedian(y / np.maximum(x, 1e-6)))
    if not np.isfinite(base_slope) or base_slope <= 0:
        base_slope = 1.0
    slope_low = _safe_slope(float(xu[1] - xu[0]), float(yu[1] - yu[0]), fallback=base_slope)
    slope_high = _safe_slope(
        float(xu[-1] - xu[-2]),
        float(yu[-1] - yu[-2]),
        fallback=max(base_slope, slope_low),
    )
    return {
        "kind": "piecewise_monotone_safe",
        "x_knots": xu.astype(float).tolist(),
        "y_knots": yu.astype(float).tolist(),
        "slope_low": float(slope_low),
        "slope_high": float(slope_high),
    }

def apply_tail_mapper_safe(
    mapper: Mapping[str, Any],
    pred_sev: np.ndarray,
    *,
    min_std_ratio: float = 0.70,
) -> np.ndarray:
    p = np.asarray(pred_sev, dtype=float)
    p = np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    kind = str(mapper.get("kind", "identity")).lower()
    if kind == "identity":
        return p

    if kind in {"piecewise_monotone_safe", "isotonic"}:
        x = np.asarray(
            mapper.get("x_knots", mapper.get("x_thresholds", [])),
            dtype=float,
        )
        y = np.asarray(
            mapper.get("y_knots", mapper.get("y_thresholds", [])),
            dtype=float,
        )
        if len(x) < 2 or len(y) < 2:
            return p

        # In-range interpolation.
        mapped = np.interp(p, x, y)

        # Safe monotone extrapolation (no hard clipping at upper tail).
        slope_low = float(mapper.get("slope_low", _safe_slope(x[1] - x[0], y[1] - y[0], 1.0)))
        slope_high = float(
            mapper.get(
                "slope_high",
                _safe_slope(x[-1] - x[-2], y[-1] - y[-2], slope_low),
            )
        )
        lo = p < x[0]
        hi = p > x[-1]
        if lo.any():
            mapped[lo] = y[0] + slope_low * (p[lo] - x[0])
        if hi.any():
            mapped[hi] = y[-1] + slope_high * (p[hi] - x[-1])
        mapped = np.maximum(mapped, 0.0)

        # No-compression guard: avoid collapsing the distribution.
        std_in = float(np.std(p))
        std_out = float(np.std(mapped))
        if std_in > 0 and std_out < (min_std_ratio * std_in):
            mean_out = float(np.mean(mapped))
            scale = float((min_std_ratio * std_in) / max(std_out, 1e-12))
            mapped = mean_out + (mapped - mean_out) * scale
            mapped = np.maximum(mapped, 0.0)
        return np.maximum(np.nan_to_num(mapped, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    raise ValueError(f"Unknown mapper kind: {kind}")

def fit_tail_mapper(
    oof_pred_sev_pos: np.ndarray | None = None,
    y_pos: np.ndarray | None = None,
    *,
    pred_sev_pos: np.ndarray | None = None,
    y_true_pos: np.ndarray | None = None,
    min_samples: int = 150,
) -> Dict[str, Any]:
    pred_arr = pred_sev_pos if pred_sev_pos is not None else oof_pred_sev_pos
    y_arr = y_true_pos if y_true_pos is not None else y_pos
    if pred_arr is None or y_arr is None:
        raise TypeError("fit_tail_mapper expects pred_sev_pos and y_true_pos arrays.")
    return fit_tail_mapper_safe(pred_arr, y_arr, min_samples=min_samples)

def apply_tail_mapper(mapper: Mapping[str, Any] | None, pred_sev: np.ndarray) -> np.ndarray:
    if mapper is None:
        return np.maximum(np.asarray(pred_sev, dtype=float), 0.0)
    return apply_tail_mapper_safe(mapper, pred_sev)

def crossfit_tail_mapper_oof(
    *,
    pred_sev: np.ndarray,
    y_sev: np.ndarray,
    y_freq: np.ndarray,
    fold_assign: np.ndarray,
) -> np.ndarray:
    p = np.asarray(pred_sev, dtype=float)
    y = np.asarray(y_sev, dtype=float)
    f = np.asarray(y_freq, dtype=int)
    folds = np.asarray(fold_assign, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(folds)
    unique_folds = sorted(set(int(k) for k in folds[valid]))
    for fold_id in unique_folds:
        val = folds == float(fold_id)
        tr = valid & (~val)
        tr_pos = tr & (f == 1)
        if tr_pos.sum() < 50 or val.sum() == 0:
            out[val] = p[val]
            continue
        mapper = fit_tail_mapper(p[tr_pos], y[tr_pos])
        out[val] = apply_tail_mapper(mapper, p[val])
    out[~valid] = p[~valid]
    out[np.isnan(out)] = p[np.isnan(out)]
    return out

