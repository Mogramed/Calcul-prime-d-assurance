from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from insurance_pricing.data.schema import TARGET_SEV_COL
from insurance_pricing.evaluation.metrics import make_tail_weights
from insurance_pricing.features.target_encoding import (
    _apply_winsor,
    _smearing_inverse,
)

def _severity_fallback(y_sev_tr: np.ndarray, n_va: int, n_te: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(y_sev_tr, dtype=float)
    pos = pos[pos > 0]
    base = float(np.nanmean(pos)) if len(pos) else 0.0
    return np.full(n_va, base, dtype=float), np.full(n_te, base, dtype=float)

def _severity_fallback_v2(
    y_sev_tr: np.ndarray,
    n_va: int,
    n_te: int,
    *,
    p_va: Optional[np.ndarray] = None,
    p_te: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(y_sev_tr, dtype=float)
    pos = pos[pos > 0]
    base = float(np.nanmean(pos)) if len(pos) else 0.0
    m_va = np.full(n_va, base, dtype=float)
    m_te = np.full(n_te, base, dtype=float)
    if p_va is None:
        p_va = np.full(n_va, 0.05, dtype=float)
    if p_te is None:
        p_te = np.full(n_te, 0.05, dtype=float)
    prime_va = p_va * m_va
    prime_te = p_te * m_te
    return p_va, m_va, prime_va, p_te, m_te, prime_te

def _fit_catboost(
    *,
    X_tr: pd.DataFrame,
    y_freq_tr: np.ndarray,
    y_sev_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_freq_va: np.ndarray,
    y_sev_va: np.ndarray,
    X_te: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    severity_mode: str,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    f_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_seed": seed,
        "verbose": False,
    }
    f_params.update(freq_params)

    s_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1800,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_seed": seed,
        "verbose": False,
    }
    s_params.update(sev_params)

    clf = CatBoostClassifier(**f_params)
    clf.fit(
        Pool(X_tr, y_freq_tr, cat_features=cat_idx),
        eval_set=Pool(X_va, y_freq_va, cat_features=cat_idx),
    )
    p_va = clf.predict_proba(Pool(X_va, cat_features=cat_idx))[:, 1]
    p_te = clf.predict_proba(Pool(X_te, cat_features=cat_idx))[:, 1]

    pos = y_freq_tr == 1
    if int(pos.sum()) < 5:
        m_va, m_te = _severity_fallback(y_sev_tr, len(X_va), len(X_te))
        return p_va, m_va, p_te, m_te

    y_pos = y_sev_tr[pos]
    y_log = np.log1p(y_pos)
    w = make_tail_weights(y_pos) if severity_mode == "weighted_tail" else None
    reg = CatBoostRegressor(**s_params)
    y_va_log = np.log1p(np.clip(y_sev_va, 0.0, None))
    reg.fit(
        Pool(X_tr.loc[pos], y_log, cat_features=cat_idx, weight=w),
        eval_set=Pool(X_va, y_va_log, cat_features=cat_idx),
    )
    z_va = reg.predict(Pool(X_va, cat_features=cat_idx))
    z_te = reg.predict(Pool(X_te, cat_features=cat_idx))
    z_tr = reg.predict(Pool(X_tr.loc[pos], cat_features=cat_idx))
    resid = y_log - z_tr
    smear = float(np.average(np.exp(resid), weights=w)) if w is not None else float(np.mean(np.exp(resid)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    m_va = np.maximum(smear * np.exp(z_va) - 1.0, 0.0)
    m_te = np.maximum(smear * np.exp(z_te) - 1.0, 0.0)
    m_va = np.nan_to_num(m_va, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
    m_te = np.nan_to_num(m_te, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
    return p_va, m_va, p_te, m_te

def _fit_catboost_fold_v2(
    *,
    family: str,
    X_tr: pd.DataFrame,
    y_freq_tr: np.ndarray,
    y_sev_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_freq_va: np.ndarray,
    y_sev_va: np.ndarray,
    X_te: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    severity_mode: str,
    tweedie_power: float,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
    direct_params: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    f_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 4000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_seed": seed,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 100,
        "use_best_model": True,
    }
    f_params.update(freq_params)
    clf = CatBoostClassifier(**f_params)
    clf.fit(
        Pool(X_tr, y_freq_tr, cat_features=cat_idx),
        eval_set=Pool(X_va, y_freq_va, cat_features=cat_idx),
    )
    p_va = clf.predict_proba(Pool(X_va, cat_features=cat_idx))[:, 1]
    p_te = clf.predict_proba(Pool(X_te, cat_features=cat_idx))[:, 1]

    fam = family.lower()
    sev_mode = severity_mode.lower()
    if fam == "direct_tweedie":
        y_target = np.clip(y_sev_tr.astype(float), 0.0, None)
        if sev_mode == "winsorized":
            y_target = _apply_winsor(y_target, quantile=0.995)
        d_params = {
            "loss_function": f"Tweedie:variance_power={float(tweedie_power):.4f}",
            "eval_metric": "RMSE",
            "iterations": 5000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
            "od_type": "Iter",
            "od_wait": 120,
            "use_best_model": True,
        }
        d_params.update(direct_params)
        reg = CatBoostRegressor(**d_params)
        reg.fit(
            Pool(X_tr, y_target, cat_features=cat_idx),
            eval_set=Pool(X_va, np.clip(y_sev_va.astype(float), 0.0, None), cat_features=cat_idx),
        )
        prime_va = np.maximum(reg.predict(Pool(X_va, cat_features=cat_idx)), 0.0)
        prime_te = np.maximum(reg.predict(Pool(X_te, cat_features=cat_idx)), 0.0)
        m_va = np.maximum(prime_va / np.clip(p_va, 1e-4, None), 0.0)
        m_te = np.maximum(prime_te / np.clip(p_te, 1e-4, None), 0.0)
        return p_va, m_va, prime_va, p_te, m_te, prime_te

    pos = y_freq_tr == 1
    if int(pos.sum()) < 20:
        return _severity_fallback_v2(y_sev_tr, len(X_va), len(X_te), p_va=p_va, p_te=p_te)

    y_pos = y_sev_tr[pos].astype(float)
    y_pos_fit = _apply_winsor(y_pos, quantile=0.995) if sev_mode == "winsorized" else y_pos
    w = make_tail_weights(y_pos_fit) if sev_mode == "weighted_tail" else None

    if fam == "two_part_tweedie":
        s_params = {
            "loss_function": f"Tweedie:variance_power={float(tweedie_power):.4f}",
            "eval_metric": "RMSE",
            "iterations": 5000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
            "od_type": "Iter",
            "od_wait": 120,
            "use_best_model": True,
        }
        s_params.update(sev_params)
        reg = CatBoostRegressor(**s_params)
        reg.fit(
            Pool(X_tr.loc[pos], y_pos_fit, cat_features=cat_idx, weight=w),
            eval_set=Pool(X_va, np.clip(y_sev_va.astype(float), 0.0, None), cat_features=cat_idx),
        )
        m_va = np.maximum(reg.predict(Pool(X_va, cat_features=cat_idx)), 0.0)
        m_te = np.maximum(reg.predict(Pool(X_te, cat_features=cat_idx)), 0.0)
    else:
        s_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 5000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
            "od_type": "Iter",
            "od_wait": 120,
            "use_best_model": True,
        }
        s_params.update(sev_params)
        reg = CatBoostRegressor(**s_params)
        y_log = np.log1p(y_pos_fit)
        y_va_log = np.log1p(np.clip(y_sev_va.astype(float), 0.0, None))
        reg.fit(
            Pool(X_tr.loc[pos], y_log, cat_features=cat_idx, weight=w),
            eval_set=Pool(X_va, y_va_log, cat_features=cat_idx),
        )
        z_va = reg.predict(Pool(X_va, cat_features=cat_idx))
        z_te = reg.predict(Pool(X_te, cat_features=cat_idx))
        z_tr = reg.predict(Pool(X_tr.loc[pos], cat_features=cat_idx))
        m_va, m_te = _smearing_inverse(y_pos_fit, z_tr=z_tr, z_va=z_va, z_te=z_te, sample_weight=w)

    prime_va = np.maximum(p_va * m_va, 0.0)
    prime_te = np.maximum(p_te * m_te, 0.0)
    return p_va, m_va, prime_va, p_te, m_te, prime_te

def _fit_catboost_fulltrain_v2(
    *,
    X_train: pd.DataFrame,
    y_freq_train: np.ndarray,
    y_sev_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    family: str,
    severity_mode: str,
    tweedie_power: float,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
    direct_params: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    fp = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_seed": seed,
        "verbose": False,
    }
    fp.update(freq_params)
    clf = CatBoostClassifier(**fp)
    clf.fit(Pool(X_train, y_freq_train, cat_features=cat_idx))
    p_te = clf.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]

    fam = family.lower()
    sev_mode = severity_mode.lower()

    if fam == "direct_tweedie":
        y_target = np.clip(y_sev_train.astype(float), 0.0, None)
        if sev_mode == "winsorized":
            y_target = _apply_winsor(y_target, quantile=0.995)
        dp = {
            "loss_function": f"Tweedie:variance_power={float(tweedie_power):.4f}",
            "eval_metric": "RMSE",
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
        }
        dp.update(direct_params)
        reg = CatBoostRegressor(**dp)
        reg.fit(Pool(X_train, y_target, cat_features=cat_idx))
        prime_te = np.maximum(reg.predict(Pool(X_test, cat_features=cat_idx)), 0.0)
        m_te = np.maximum(prime_te / np.clip(p_te, 1e-4, None), 0.0)
        return p_te, m_te, prime_te

    pos = y_freq_train == 1
    if int(pos.sum()) < 20:
        _, m_te = _severity_fallback(y_sev_train, 1, len(X_test))
        prime_te = np.maximum(p_te * m_te, 0.0)
        return p_te, m_te, prime_te

    y_pos = y_sev_train[pos].astype(float)
    y_pos_fit = _apply_winsor(y_pos, quantile=0.995) if sev_mode == "winsorized" else y_pos
    w = make_tail_weights(y_pos_fit) if sev_mode == "weighted_tail" else None

    if fam == "two_part_tweedie":
        sp = {
            "loss_function": f"Tweedie:variance_power={float(tweedie_power):.4f}",
            "eval_metric": "RMSE",
            "iterations": 3500,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
        }
        sp.update(sev_params)
        reg = CatBoostRegressor(**sp)
        reg.fit(Pool(X_train.loc[pos], y_pos_fit, cat_features=cat_idx, weight=w))
        m_te = np.maximum(reg.predict(Pool(X_test, cat_features=cat_idx)), 0.0)
    else:
        sp = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 3500,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
        }
        sp.update(sev_params)
        reg = CatBoostRegressor(**sp)
        y_log = np.log1p(y_pos_fit)
        reg.fit(Pool(X_train.loc[pos], y_log, cat_features=cat_idx, weight=w))
        z_te = reg.predict(Pool(X_test, cat_features=cat_idx))
        z_tr = reg.predict(Pool(X_train.loc[pos], cat_features=cat_idx))
        _, m_te = _smearing_inverse(y_pos_fit, z_tr=z_tr, z_va=z_te, z_te=z_te, sample_weight=w)

    m_te = np.maximum(np.nan_to_num(m_te, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    prime_te = np.maximum(p_te * m_te, 0.0)
    return p_te, m_te, prime_te

