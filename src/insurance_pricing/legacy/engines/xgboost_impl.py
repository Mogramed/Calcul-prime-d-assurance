from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from insurance_pricing._typing import FloatArray, IntArray
from insurance_pricing.data.schema import OrdinalFrameEncoder
from insurance_pricing.evaluation.metrics import make_tail_weights
from insurance_pricing.features.target_encoding import (
    _apply_winsor,
    _smearing_inverse,
)

from .catboost_impl import _severity_fallback, _severity_fallback_v2


def _fit_xgb(
    *,
    X_tr: pd.DataFrame,
    y_freq_tr: IntArray,
    y_sev_tr: FloatArray,
    X_va: pd.DataFrame,
    y_freq_va: IntArray,
    y_sev_va: FloatArray,
    X_te: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    severity_mode: str,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    from xgboost import XGBClassifier, XGBRegressor

    enc = OrdinalFrameEncoder(cat_cols).fit(X_tr)
    Xtr = enc.transform(X_tr)
    Xva = enc.transform(X_va)
    Xte = enc.transform(X_te)

    f_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 1800,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    f_params.update(freq_params)

    s_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2200,
        "learning_rate": 0.03,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    s_params.update(sev_params)

    clf = XGBClassifier(**f_params)
    clf.fit(Xtr, y_freq_tr, verbose=False)
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    pos = y_freq_tr == 1
    if int(pos.sum()) < 5:
        m_va, m_te = _severity_fallback(y_sev_tr, len(X_va), len(X_te))
        return p_va, m_va, p_te, m_te

    y_pos = y_sev_tr[pos]
    y_log = np.log1p(y_pos)
    w = make_tail_weights(y_pos) if severity_mode == "weighted_tail" else None
    reg = XGBRegressor(**s_params)
    reg.fit(Xtr.loc[pos], y_log, sample_weight=w, verbose=False)
    z_va = reg.predict(Xva)
    z_te = reg.predict(Xte)
    z_tr = reg.predict(Xtr.loc[pos])
    resid = y_log - z_tr
    smear = (
        float(np.average(np.exp(resid), weights=w))
        if w is not None
        else float(np.mean(np.exp(resid)))
    )
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    m_va = np.maximum(smear * np.exp(z_va) - 1.0, 0.0)
    m_te = np.maximum(smear * np.exp(z_te) - 1.0, 0.0)
    m_va = np.nan_to_num(
        m_va, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0
    )
    m_te = np.nan_to_num(
        m_te, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0
    )
    return p_va, m_va, p_te, m_te


def _fit_xgb_fold_v2(
    *,
    family: str,
    X_tr: pd.DataFrame,
    y_freq_tr: IntArray,
    y_sev_tr: FloatArray,
    X_va: pd.DataFrame,
    y_freq_va: IntArray,
    y_sev_va: FloatArray,
    X_te: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    severity_mode: str,
    tweedie_power: float,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
    direct_params: Mapping[str, Any],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    from xgboost import XGBClassifier, XGBRegressor

    enc = OrdinalFrameEncoder(cat_cols).fit(X_tr)
    Xtr = enc.transform(X_tr)
    Xva = enc.transform(X_va)
    Xte = enc.transform(X_te)

    f_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 6000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
    }
    f_params.update(freq_params)
    clf = XGBClassifier(**f_params)
    clf.fit(Xtr, y_freq_tr, eval_set=[(Xva, y_freq_va)], verbose=False)
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    fam = family.lower()
    sev_mode = severity_mode.lower()
    if fam == "direct_tweedie":
        y_target = np.clip(y_sev_tr.astype(float), 0.0, None)
        if sev_mode == "winsorized":
            y_target = _apply_winsor(y_target, quantile=0.995)
        d_params = {
            "objective": "reg:tweedie",
            "eval_metric": "rmse",
            "tweedie_variance_power": float(tweedie_power),
            "n_estimators": 7000,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "early_stopping_rounds": 120,
        }
        d_params.update(direct_params)
        reg = XGBRegressor(**d_params)
        reg.fit(
            Xtr,
            y_target,
            eval_set=[(Xva, np.clip(y_sev_va.astype(float), 0.0, None))],
            verbose=False,
        )
        prime_va = np.maximum(reg.predict(Xva), 0.0)
        prime_te = np.maximum(reg.predict(Xte), 0.0)
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
            "objective": "reg:tweedie",
            "eval_metric": "rmse",
            "tweedie_variance_power": float(tweedie_power),
            "n_estimators": 7000,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "early_stopping_rounds": 120,
        }
        s_params.update(sev_params)
        reg = XGBRegressor(**s_params)
        reg.fit(
            Xtr.loc[pos],
            y_pos_fit,
            sample_weight=w,
            eval_set=[(Xva, np.clip(y_sev_va.astype(float), 0.0, None))],
            verbose=False,
        )
        m_va = np.maximum(reg.predict(Xva), 0.0)
        m_te = np.maximum(reg.predict(Xte), 0.0)
    else:
        s_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 7000,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "early_stopping_rounds": 120,
        }
        s_params.update(sev_params)
        reg = XGBRegressor(**s_params)
        y_log = np.log1p(y_pos_fit)
        y_va_log = np.log1p(np.clip(y_sev_va.astype(float), 0.0, None))
        reg.fit(
            Xtr.loc[pos],
            y_log,
            sample_weight=w,
            eval_set=[(Xva, y_va_log)],
            verbose=False,
        )
        z_va = reg.predict(Xva)
        z_te = reg.predict(Xte)
        z_tr = reg.predict(Xtr.loc[pos])
        m_va, m_te = _smearing_inverse(y_pos_fit, z_tr=z_tr, z_va=z_va, z_te=z_te, sample_weight=w)

    prime_va = np.maximum(p_va * m_va, 0.0)
    prime_te = np.maximum(p_te * m_te, 0.0)
    return p_va, m_va, prime_va, p_te, m_te, prime_te


def _fit_xgb_fulltrain_v2(
    *,
    X_train: pd.DataFrame,
    y_freq_train: IntArray,
    y_sev_train: FloatArray,
    X_test: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    family: str,
    severity_mode: str,
    tweedie_power: float,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
    direct_params: Mapping[str, Any],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    from xgboost import XGBClassifier, XGBRegressor

    enc = OrdinalFrameEncoder(cat_cols).fit(X_train)
    Xtr = enc.transform(X_train)
    Xte = enc.transform(X_test)

    fp = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    fp.update(freq_params)
    fp.pop("early_stopping_rounds", None)
    clf = XGBClassifier(**fp)
    clf.fit(Xtr, y_freq_train, verbose=False)
    p_te = clf.predict_proba(Xte)[:, 1]

    fam = family.lower()
    sev_mode = severity_mode.lower()

    if fam == "direct_tweedie":
        y_target = np.clip(y_sev_train.astype(float), 0.0, None)
        if sev_mode == "winsorized":
            y_target = _apply_winsor(y_target, quantile=0.995)
        dp = {
            "objective": "reg:tweedie",
            "eval_metric": "rmse",
            "tweedie_variance_power": float(tweedie_power),
            "n_estimators": 3500,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        dp.update(direct_params)
        dp.pop("early_stopping_rounds", None)
        reg = XGBRegressor(**dp)
        reg.fit(Xtr, y_target, verbose=False)
        prime_te = np.maximum(reg.predict(Xte), 0.0)
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
            "objective": "reg:tweedie",
            "eval_metric": "rmse",
            "tweedie_variance_power": float(tweedie_power),
            "n_estimators": 4000,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        sp.update(sev_params)
        sp.pop("early_stopping_rounds", None)
        reg = XGBRegressor(**sp)
        reg.fit(Xtr.loc[pos], y_pos_fit, sample_weight=w, verbose=False)
        m_te = np.maximum(reg.predict(Xte), 0.0)
    else:
        sp = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 4000,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        sp.update(sev_params)
        sp.pop("early_stopping_rounds", None)
        reg = XGBRegressor(**sp)
        y_log = np.log1p(y_pos_fit)
        reg.fit(Xtr.loc[pos], y_log, sample_weight=w, verbose=False)
        z_te = reg.predict(Xte)
        z_tr = reg.predict(Xtr.loc[pos])
        _, m_te = _smearing_inverse(y_pos_fit, z_tr=z_tr, z_va=z_te, z_te=z_te, sample_weight=w)

    m_te = np.maximum(np.nan_to_num(m_te, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    prime_te = np.maximum(p_te * m_te, 0.0)
    return p_te, m_te, prime_te
