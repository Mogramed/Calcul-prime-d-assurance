from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def summarize_prime_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    top1_thr = float(np.quantile(y, 0.99))
    m_top = y >= top1_thr
    return {
        "rmse_prime": rmse(y, p),
        "mae_prime": mae(y, p),
        "rmse_prime_top1pct": rmse(y[m_top], p[m_top]) if np.any(m_top) else float("nan"),
    }


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def make_tail_weights(y_pos: np.ndarray) -> np.ndarray:
    y = np.asarray(y_pos, dtype=float)
    ref = max(float(np.nanpercentile(y, 50)), 1.0)
    w = np.sqrt((y + 1.0) / (ref + 1.0))
    q90 = float(np.nanpercentile(y, 90))
    w[y >= q90] *= 1.5
    return np.clip(w, 1.0, 8.0)


def compute_metric_row(
    *,
    y_freq_true: np.ndarray,
    y_sev_true: np.ndarray,
    pred_freq: np.ndarray,
    pred_sev: np.ndarray,
    pred_prime: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pred_freq = np.nan_to_num(np.asarray(pred_freq, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    pred_sev = np.nan_to_num(np.asarray(pred_sev, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_freq_true = np.asarray(y_freq_true, dtype=int)
    y_sev_true = np.asarray(y_sev_true, dtype=float)

    if pred_prime is None:
        pred_prime = pred_freq * pred_sev
    else:
        pred_prime = np.nan_to_num(
            np.asarray(pred_prime, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
    pos = y_freq_true == 1
    q99_true = float(np.nanpercentile(y_sev_true[pos], 99)) if pos.any() else float("nan")
    q99_pred = float(np.nanpercentile(pred_sev[pos], 99)) if pos.any() else float("nan")
    return {
        "rmse_prime": rmse(y_sev_true, pred_prime),
        "auc_freq": _safe_auc(y_freq_true, pred_freq),
        "brier_freq": float(brier_score_loss(y_freq_true, pred_freq)),
        "rmse_sev_pos": rmse(y_sev_true[pos], pred_sev[pos]) if pos.any() else float("nan"),
        "q99_ratio_pos": (q99_pred / q99_true) if pos.any() and q99_true > 0 else float("nan"),
    }

