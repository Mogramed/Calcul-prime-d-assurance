from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_fit_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    method = kwargs.get("method")
    if "p_pred" in kwargs and "y_true" in kwargs:
        probs = kwargs["p_pred"]
        y_true = kwargs["y_true"]
    elif "probs" in kwargs and "y_true" in kwargs:
        probs = kwargs["probs"]
        y_true = kwargs["y_true"]
    elif len(args) >= 3:
        probs, y_true, method = args[0], args[1], args[2]
    elif len(args) >= 2:
        probs, y_true = args[0], args[1]
    else:
        raise TypeError("fit_calibrator expects (probs, y_true, method) or keywords y_true=, p_pred=, method=.")
    if method is None:
        raise TypeError("fit_calibrator requires `method`.")
    return np.asarray(probs, dtype=float), np.asarray(y_true, dtype=int), str(method)


def fit_calibrator(*args: Any, **kwargs: Any):
    probs, y_true, method = _resolve_fit_inputs(args, kwargs)
    m = method.lower()
    if m == "none":
        return None
    if m == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probs, y_true)
        return model
    if m == "platt":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=2000)
        model.fit(probs.reshape(-1, 1), y_true)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def _resolve_apply_inputs(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[np.ndarray, str]:
    method = kwargs.get("method")
    if "p_pred" in kwargs:
        probs = kwargs["p_pred"]
    elif "probs" in kwargs:
        probs = kwargs["probs"]
    elif len(args) >= 2:
        probs, method = args[0], args[1]
    elif len(args) == 1:
        probs = args[0]
    else:
        raise TypeError("apply_calibrator expects (model, probs, method) or keywords p_pred=, method=.")
    if method is None:
        raise TypeError("apply_calibrator requires `method`.")
    return np.asarray(probs, dtype=float), str(method)


def apply_calibrator(model, *args: Any, **kwargs: Any) -> np.ndarray:
    probs, method = _resolve_apply_inputs(model, args, kwargs)
    m = method.lower()
    if m == "none" or model is None:
        return probs
    if m == "isotonic":
        return model.transform(probs)
    if m == "platt":
        return model.predict_proba(probs.reshape(-1, 1))[:, 1]
    raise ValueError(f"Unknown calibration method: {method}")

def crossfit_calibrate_oof(
    *,
    probs: np.ndarray,
    y_true: np.ndarray,
    fold_assign: np.ndarray,
    method: str,
) -> np.ndarray:
    m = method.lower()
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int)
    folds = np.asarray(fold_assign, dtype=float)
    if m == "none":
        return p.copy()

    out = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(folds)
    unique_folds = sorted(set(int(f) for f in folds[valid]))
    for f in unique_folds:
        val = folds == float(f)
        tr = valid & (~val)
        if tr.sum() == 0 or val.sum() == 0:
            out[val] = p[val]
            continue
        c = fit_calibrator(p[tr], y[tr], m)
        out[val] = apply_calibrator(c, p[val], m)
    out[~valid] = p[~valid]
    missing = np.isnan(out) & valid
    out[missing] = p[missing]
    return out

