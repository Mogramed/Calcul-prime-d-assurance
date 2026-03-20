from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from insurance_pricing.data.schema import DatasetBundle, INDEX_COL
from insurance_pricing.features.target_encoding import _add_fold_target_encoding
from insurance_pricing.legacy.engines.catboost_impl import _fit_catboost_fulltrain_v2
from insurance_pricing.legacy.engines.lightgbm_impl import _fit_lgbm_fulltrain_v2
from insurance_pricing.legacy.engines.xgboost_impl import _fit_xgb_fulltrain_v2
from insurance_pricing.training.benchmark import _fit_predict_fold_v2

def fit_full_two_part_predict(
    *,
    engine: str,
    X_train: pd.DataFrame,
    y_freq_train: np.ndarray,
    y_sev_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    severity_mode: str,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Train on full train and return (test_freq_raw, test_sev)."""
    e = engine.lower()

    if e == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool

        cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
        fp = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "iterations": 1200,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
        }
        fp.update(freq_params)
        sp = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 1800,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "random_seed": seed,
            "verbose": False,
        }
        sp.update(sev_params)

        clf = CatBoostClassifier(**fp)
        clf.fit(Pool(X_train, y_freq_train, cat_features=cat_idx))
        p_te = clf.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]

        pos = y_freq_train == 1
        if int(pos.sum()) < 5:
            _, m_te = _severity_fallback(y_sev_train, 1, len(X_test))
            return p_te, m_te
        y_pos = y_sev_train[pos]
        y_log = np.log1p(y_pos)
        w = make_tail_weights(y_pos) if severity_mode == "weighted_tail" else None
        reg = CatBoostRegressor(**sp)
        reg.fit(Pool(X_train.loc[pos], y_log, cat_features=cat_idx, weight=w))
        z_te = reg.predict(Pool(X_test, cat_features=cat_idx))
        z_tr = reg.predict(Pool(X_train.loc[pos], cat_features=cat_idx))
        resid = y_log - z_tr
        smear = (
            float(np.average(np.exp(resid), weights=w))
            if w is not None
            else float(np.mean(np.exp(resid)))
        )
        if not np.isfinite(smear) or smear <= 0:
            smear = 1.0
        m_te = np.maximum(smear * np.exp(z_te) - 1.0, 0.0)
        m_te = np.nan_to_num(m_te, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
        return p_te, m_te

    if e in {"lightgbm", "xgboost"}:
        if e == "lightgbm":
            from lightgbm import LGBMClassifier, LGBMRegressor
        else:
            from xgboost import XGBClassifier, XGBRegressor

        enc = OrdinalFrameEncoder(cat_cols).fit(X_train)
        Xtr = enc.transform(X_train)
        Xte = enc.transform(X_test)

        if e == "lightgbm":
            fp = {
                "objective": "binary",
                "n_estimators": 2000,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed,
                "n_jobs": -1,
            }
            sp = {
                "objective": "rmse",
                "n_estimators": 2500,
                "learning_rate": 0.03,
                "num_leaves": 127,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed,
                "n_jobs": -1,
            }
            fp.update(freq_params)
            sp.update(sev_params)
            clf = LGBMClassifier(**fp)
            reg = LGBMRegressor(**sp)
        else:
            fp = {
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
            sp = {
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
            fp.update(freq_params)
            sp.update(sev_params)
            clf = XGBClassifier(**fp)
            reg = XGBRegressor(**sp)

        if e == "xgboost":
            clf.fit(Xtr, y_freq_train, verbose=False)
        else:
            clf.fit(Xtr, y_freq_train)
        p_te = clf.predict_proba(Xte)[:, 1]

        pos = y_freq_train == 1
        if int(pos.sum()) < 5:
            _, m_te = _severity_fallback(y_sev_train, 1, len(X_test))
            return p_te, m_te
        y_pos = y_sev_train[pos]
        y_log = np.log1p(y_pos)
        w = make_tail_weights(y_pos) if severity_mode == "weighted_tail" else None
        if e == "xgboost":
            reg.fit(Xtr.loc[pos], y_log, sample_weight=w, verbose=False)
        else:
            reg.fit(Xtr.loc[pos], y_log, sample_weight=w)
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
        m_te = np.maximum(smear * np.exp(z_te) - 1.0, 0.0)
        m_te = np.nan_to_num(m_te, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
        return p_te, m_te

    raise ValueError(f"Unsupported engine: {engine}")

def fit_full_predict(
    *,
    spec: Mapping[str, Any],
    bundle: DatasetBundle,
    seed: int,
    valid_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    n = len(bundle.X_train)
    n_val = max(int(n * valid_ratio), 1000)
    order = np.argsort(bundle.train_raw[INDEX_COL].to_numpy())
    tr_idx = order[:-n_val]
    va_idx = order[-n_val:]
    X_tr = bundle.X_train.iloc[tr_idx].copy()
    X_va = bundle.X_train.iloc[va_idx].copy()
    X_te = bundle.X_test.copy()
    y_freq = bundle.y_freq.to_numpy(dtype=int)
    y_sev = bundle.y_sev.to_numpy(dtype=float)
    y_freq_tr = y_freq[tr_idx]
    y_sev_tr = y_sev[tr_idx]
    y_freq_va = y_freq[va_idx]
    y_sev_va = y_sev[va_idx]

    if bool(spec.get("use_target_encoding", False)):
        X_tr, X_va, X_te = _add_fold_target_encoding(
            X_tr=X_tr,
            y_freq_tr=y_freq_tr,
            y_sev_tr=y_sev_tr,
            X_va=X_va,
            X_te=X_te,
            cols=list(spec.get("target_encode_cols", [])),
            smoothing=float(spec.get("target_encoding_smoothing", 20.0)),
        )
    cat_cols_fold = [c for c in bundle.cat_cols if c in X_tr.columns and c in X_va.columns]
    p_va, m_va, prime_va, p_te, m_te, prime_te = _fit_predict_fold_v2(
        engine=str(spec.get("engine", "catboost")),
        family=str(spec.get("family", "two_part_classic")),
        X_tr=X_tr,
        y_freq_tr=y_freq_tr,
        y_sev_tr=y_sev_tr,
        X_va=X_va,
        y_freq_va=y_freq_va,
        y_sev_va=y_sev_va,
        X_te=X_te,
        cat_cols=cat_cols_fold,
        seed=int(seed),
        severity_mode=str(spec.get("severity_mode", "classic")),
        tweedie_power=float(spec.get("tweedie_power", 1.5)),
        freq_params=dict(spec.get("freq_params", {})),
        sev_params=dict(spec.get("sev_params", {})),
        direct_params=dict(spec.get("direct_params", {})),
    )
    return {
        "valid_freq": np.clip(np.nan_to_num(p_va, nan=0.0), 0.0, 1.0),
        "valid_sev": np.maximum(np.nan_to_num(m_va, nan=0.0), 0.0),
        "valid_prime": np.maximum(np.nan_to_num(prime_va, nan=0.0), 0.0),
        "test_freq": np.clip(np.nan_to_num(p_te, nan=0.0), 0.0, 1.0),
        "test_sev": np.maximum(np.nan_to_num(m_te, nan=0.0), 0.0),
        "test_prime": np.maximum(np.nan_to_num(prime_te, nan=0.0), 0.0),
    }

def _apply_fulltrain_complexity(
    engine: str,
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
    direct_params: Mapping[str, Any],
    complexity: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    fp = dict(freq_params)
    sp = dict(sev_params)
    dp = dict(direct_params)
    if not complexity:
        return fp, sp, dp

    key = "iterations" if engine.lower() == "catboost" else "n_estimators"
    c = dict(complexity)

    if "freq_iterations" in c:
        fp[key] = int(c["freq_iterations"])
    if "sev_iterations" in c:
        sp[key] = int(c["sev_iterations"])
    if "direct_iterations" in c:
        dp[key] = int(c["direct_iterations"])

    if "freq_n_estimators" in c:
        fp[key] = int(c["freq_n_estimators"])
    if "sev_n_estimators" in c:
        sp[key] = int(c["sev_n_estimators"])
    if "direct_n_estimators" in c:
        dp[key] = int(c["direct_n_estimators"])

    if "iterations" in c:
        fp[key] = int(c["iterations"])
        sp[key] = int(c["iterations"])
        dp[key] = int(c["iterations"])

    return fp, sp, dp

def fit_full_predict_fulltrain(
    *,
    spec: Mapping[str, Any],
    bundle: DatasetBundle,
    seed: int,
    complexity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    engine = str(spec.get("engine", "catboost")).lower()
    family = str(spec.get("family", "two_part_classic")).lower()
    severity_mode = str(spec.get("severity_mode", "classic")).lower()
    tweedie_power = float(spec.get("tweedie_power", 1.5))
    freq_params = dict(spec.get("freq_params", {}))
    sev_params = dict(spec.get("sev_params", {}))
    direct_params = dict(spec.get("direct_params", {}))
    fulltrain_complexity = dict(spec.get("fulltrain_complexity", {}))
    if complexity:
        fulltrain_complexity.update(dict(complexity))

    freq_params, sev_params, direct_params = _apply_fulltrain_complexity(
        engine=engine,
        freq_params=freq_params,
        sev_params=sev_params,
        direct_params=direct_params,
        complexity=fulltrain_complexity,
    )

    X_train = bundle.X_train.copy()
    X_test = bundle.X_test.copy()
    y_freq = bundle.y_freq.to_numpy(dtype=int)
    y_sev = bundle.y_sev.to_numpy(dtype=float)

    if bool(spec.get("use_target_encoding", False)):
        X_train, _, X_test = _add_fold_target_encoding(
            X_tr=X_train,
            y_freq_tr=y_freq,
            y_sev_tr=y_sev,
            X_va=X_train.copy(),
            X_te=X_test,
            cols=list(spec.get("target_encode_cols", [])),
            smoothing=float(spec.get("target_encoding_smoothing", 20.0)),
        )
    cat_cols = [c for c in bundle.cat_cols if c in X_train.columns]

    if engine == "catboost":
        p_te, m_te, prime_te = _fit_catboost_fulltrain_v2(
            X_train=X_train,
            y_freq_train=y_freq,
            y_sev_train=y_sev,
            X_test=X_test,
            cat_cols=cat_cols,
            seed=seed,
            family=family,
            severity_mode=severity_mode,
            tweedie_power=tweedie_power,
            freq_params=freq_params,
            sev_params=sev_params,
            direct_params=direct_params,
        )
    elif engine == "lightgbm":
        p_te, m_te, prime_te = _fit_lgbm_fulltrain_v2(
            X_train=X_train,
            y_freq_train=y_freq,
            y_sev_train=y_sev,
            X_test=X_test,
            cat_cols=cat_cols,
            seed=seed,
            family=family,
            severity_mode=severity_mode,
            tweedie_power=tweedie_power,
            freq_params=freq_params,
            sev_params=sev_params,
            direct_params=direct_params,
        )
    elif engine == "xgboost":
        p_te, m_te, prime_te = _fit_xgb_fulltrain_v2(
            X_train=X_train,
            y_freq_train=y_freq,
            y_sev_train=y_sev,
            X_test=X_test,
            cat_cols=cat_cols,
            seed=seed,
            family=family,
            severity_mode=severity_mode,
            tweedie_power=tweedie_power,
            freq_params=freq_params,
            sev_params=sev_params,
            direct_params=direct_params,
        )
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    return {
        "test_freq": np.clip(np.nan_to_num(p_te, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
        "test_sev": np.maximum(np.nan_to_num(m_te, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
        "test_prime": np.maximum(np.nan_to_num(prime_te, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
    }

