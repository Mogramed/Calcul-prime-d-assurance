from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.insurance_pricing.data.io import ensure_dir
from src.insurance_pricing.data.schema import DatasetBundle, INDEX_COL
from src.insurance_pricing.evaluation.diagnostics import (
    build_prediction_distribution_table,
    compute_prediction_distribution_audit,
)
from src.insurance_pricing.evaluation.metrics import compute_metric_row, rmse
from src.insurance_pricing.features.target_encoding import _add_fold_target_encoding
from src.insurance_pricing.models.calibration import (
    apply_calibrator,
    crossfit_calibrate_oof,
    fit_calibrator,
)
from src.insurance_pricing.legacy.engines.catboost_impl import (
    _fit_catboost,
    _fit_catboost_fold_v2,
)
from src.insurance_pricing.legacy.engines.lightgbm_impl import (
    _fit_lgbm,
    _fit_lgbm_fold_v2,
)
from src.insurance_pricing.legacy.engines.xgboost_impl import _fit_xgb, _fit_xgb_fold_v2
from src.insurance_pricing.models.tail import (
    apply_tail_mapper,
    crossfit_tail_mapper_oof,
    fit_tail_mapper,
)
from src.insurance_pricing.training.selection import optimize_non_negative_weights


def make_run_id(df: pd.DataFrame) -> pd.Series:
    if "run_id" in df.columns:
        return df["run_id"].astype(str)

    cols_new = [
        "feature_set",
        "engine",
        "family",
        "tweedie_power",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
    ]
    if all(c in df.columns for c in cols_new):
        out = df[cols_new[0]].astype(str)
        for c in cols_new[1:]:
            out = out + "|" + df[c].astype(str)
        return out

    cols_legacy_v2 = [
        "feature_set",
        "engine",
        "family",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
    ]
    if all(c in df.columns for c in cols_legacy_v2):
        out = df[cols_legacy_v2[0]].astype(str)
        for c in cols_legacy_v2[1:]:
            out = out + "|" + df[c].astype(str)
        return out

    cols_v1 = ["engine", "config_id", "seed", "severity_mode", "calibration"]
    if all(c in df.columns for c in cols_v1):
        out = df[cols_v1[0]].astype(str)
        for c in cols_v1[1:]:
            out = out + "|" + df[c].astype(str)
        return out

    raise KeyError("Cannot build run_id: required columns are missing.")


def _attach_run_health_columns(
    run_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if run_df.empty:
        return run_df, pd.DataFrame()
    rr = run_df.copy()
    if "run_id" not in rr.columns:
        rr["run_id"] = make_run_id(rr)

    dist = build_prediction_distribution_table(pred_df)
    if dist.empty:
        for c in [
            "rmse_gap_secondary",
            "rmse_gap_aux",
            "tail_dispersion_flag",
            "distribution_collapse_flag",
        ]:
            rr[c] = np.nan
        return rr, dist

    piv_rmse = (
        rr[rr["level"] == "run"]
        .pivot_table(index="run_id", columns="split", values="rmse_prime", aggfunc="mean")
        .rename_axis(None, axis=1)
    )
    rmse_primary = piv_rmse["primary_time"] if "primary_time" in piv_rmse.columns else pd.Series(dtype=float)
    rmse_secondary = piv_rmse["secondary_group"] if "secondary_group" in piv_rmse.columns else pd.Series(dtype=float)
    rmse_aux = piv_rmse["aux_blocked5"] if "aux_blocked5" in piv_rmse.columns else pd.Series(dtype=float)
    gap_secondary = (rmse_secondary - rmse_primary).to_dict()
    gap_aux = (rmse_aux - rmse_primary).to_dict()

    dist_test = dist[dist["sample"] == "test"].copy()
    dist_oof = dist[dist["sample"] == "oof"].copy()
    q99_oof_primary = (
        dist_oof[dist_oof["split"] == "primary_time"]
        .set_index("run_id")["pred_q99"]
        .to_dict()
    )
    test_stats = (
        dist_test.set_index(["run_id", "split"])[
            ["pred_q90", "pred_q99", "pred_std", "distribution_collapse_flag"]
        ]
        if not dist_test.empty
        else pd.DataFrame()
    )

    rr["rmse_gap_secondary"] = rr["run_id"].map(gap_secondary).astype(float)
    rr["rmse_gap_aux"] = rr["run_id"].map(gap_aux).astype(float)
    rr["pred_q99_test"] = rr.apply(
        lambda r: float(
            test_stats.loc[(r["run_id"], r["split"]), "pred_q99"]
            if (r["run_id"], r["split"]) in test_stats.index
            else np.nan
        ),
        axis=1,
    )
    rr["pred_q90_test"] = rr.apply(
        lambda r: float(
            test_stats.loc[(r["run_id"], r["split"]), "pred_q90"]
            if (r["run_id"], r["split"]) in test_stats.index
            else np.nan
        ),
        axis=1,
    )
    rr["pred_q99_oof_primary"] = rr["run_id"].map(q99_oof_primary).astype(float)
    rr["q99_test_over_oof_primary"] = rr["pred_q99_test"] / rr["pred_q99_oof_primary"].clip(lower=1e-9)
    rr["distribution_collapse_flag"] = rr.apply(
        lambda r: int(
            (
                np.isfinite(r["pred_q90_test"])
                and np.isfinite(r["pred_q99_test"])
                and (r["pred_q99_test"] <= (1.02 * r["pred_q90_test"]))
            )
            or (
                np.isfinite(r["q99_test_over_oof_primary"])
                and (r["q99_test_over_oof_primary"] < 0.60)
            )
            or (
                (r["run_id"], r["split"]) in test_stats.index
                and int(test_stats.loc[(r["run_id"], r["split"]), "distribution_collapse_flag"]) == 1
            )
        ),
        axis=1,
    )
    rr["tail_dispersion_flag"] = rr.apply(
        lambda r: int(
            (np.isfinite(r.get("q99_ratio_pos", np.nan)) and r["q99_ratio_pos"] < 0.35)
            or (
                np.isfinite(r["q99_test_over_oof_primary"])
                and (r["q99_test_over_oof_primary"] < 0.75)
            )
        ),
        axis=1,
    )
    return rr, dist

def fit_predict_two_part(
    *,
    engine: str,
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
    e = engine.lower()
    if e == "catboost":
        return _fit_catboost(
            X_tr=X_tr,
            y_freq_tr=y_freq_tr,
            y_sev_tr=y_sev_tr,
            X_va=X_va,
            y_freq_va=y_freq_va,
            y_sev_va=y_sev_va,
            X_te=X_te,
            cat_cols=cat_cols,
            seed=seed,
            severity_mode=severity_mode,
            freq_params=freq_params,
            sev_params=sev_params,
        )
    if e == "lightgbm":
        return _fit_lgbm(
            X_tr=X_tr,
            y_freq_tr=y_freq_tr,
            y_sev_tr=y_sev_tr,
            X_va=X_va,
            y_freq_va=y_freq_va,
            y_sev_va=y_sev_va,
            X_te=X_te,
            cat_cols=cat_cols,
            seed=seed,
            severity_mode=severity_mode,
            freq_params=freq_params,
            sev_params=sev_params,
        )
    if e == "xgboost":
        return _fit_xgb(
            X_tr=X_tr,
            y_freq_tr=y_freq_tr,
            y_sev_tr=y_sev_tr,
            X_va=X_va,
            y_freq_va=y_freq_va,
            y_sev_va=y_sev_va,
            X_te=X_te,
            cat_cols=cat_cols,
            seed=seed,
            severity_mode=severity_mode,
            freq_params=freq_params,
            sev_params=sev_params,
        )
    raise ValueError(f"Unsupported engine: {engine}")

def run_cv_experiment(
    *,
    split_name: str,
    engine: str,
    config_id: str,
    X: pd.DataFrame,
    y_freq: pd.Series,
    y_sev: pd.Series,
    folds: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    X_test: pd.DataFrame,
    cat_cols: Sequence[str],
    seed: int,
    severity_mode: str,
    calibration_methods: Sequence[str],
    freq_params: Mapping[str, Any],
    sev_params: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(X)
    fold_assign = np.full(n, np.nan)
    oof_freq = np.full(n, np.nan)
    oof_sev = np.full(n, np.nan)
    y_freq_np = y_freq.to_numpy(dtype=int)
    y_sev_np = y_sev.to_numpy(dtype=float)

    test_freq_parts: List[np.ndarray] = []
    test_sev_parts: List[np.ndarray] = []
    fold_rows: List[Dict[str, Any]] = []

    for fold_id, (tr, va) in folds.items():
        p_va, m_va, p_te, m_te = fit_predict_two_part(
            engine=engine,
            X_tr=X.iloc[tr],
            y_freq_tr=y_freq_np[tr],
            y_sev_tr=y_sev_np[tr],
            X_va=X.iloc[va],
            y_freq_va=y_freq_np[va],
            y_sev_va=y_sev_np[va],
            X_te=X_test,
            cat_cols=cat_cols,
            seed=seed + int(fold_id),
            severity_mode=severity_mode,
            freq_params=freq_params,
            sev_params=sev_params,
        )
        oof_freq[va] = p_va
        oof_sev[va] = m_va
        fold_assign[va] = float(fold_id)
        test_freq_parts.append(p_te)
        test_sev_parts.append(m_te)
        m = compute_metric_row(
            y_freq_true=y_freq_np[va],
            y_sev_true=y_sev_np[va],
            pred_freq=p_va,
            pred_sev=m_va,
        )
        fold_rows.append(
            {
                "level": "fold",
                "split": split_name,
                "engine": engine,
                "config_id": config_id,
                "seed": int(seed),
                "severity_mode": severity_mode,
                "calibration": "none",
                "fold_id": int(fold_id),
                **m,
            }
        )

    valid = ~np.isnan(oof_freq)
    test_freq_mean = np.nanmean(np.vstack(test_freq_parts), axis=0)
    test_sev_mean = np.nanmean(np.vstack(test_sev_parts), axis=0)
    run_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []

    for calib in calibration_methods:
        c = calib.lower()
        if c == "none":
            oof_freq_cal = oof_freq.copy()
            test_freq_cal = test_freq_mean.copy()
        else:
            oof_freq_cal = crossfit_calibrate_oof(
                probs=oof_freq, y_true=y_freq_np, fold_assign=fold_assign, method=c
            )
            full_cal = fit_calibrator(oof_freq[valid], y_freq_np[valid], c)
            test_freq_cal = apply_calibrator(full_cal, test_freq_mean, c)

        bad = np.isnan(oof_freq_cal) & valid
        if bad.any():
            oof_freq_cal[bad] = oof_freq[bad]
        test_freq_cal = np.nan_to_num(test_freq_cal, nan=float(np.nanmean(oof_freq[valid])))

        m = compute_metric_row(
            y_freq_true=y_freq_np[valid],
            y_sev_true=y_sev_np[valid],
            pred_freq=oof_freq_cal[valid],
            pred_sev=oof_sev[valid],
        )
        run_rows.append(
            {
                "level": "run",
                "split": split_name,
                "engine": engine,
                "config_id": config_id,
                "seed": int(seed),
                "severity_mode": severity_mode,
                "calibration": c,
                "fold_id": -1,
                "n_valid": int(valid.sum()),
                **m,
            }
        )

        pred_frames.append(
            pd.DataFrame(
                {
                    "row_idx": np.arange(n, dtype=int),
                    "is_test": 0,
                    "split": split_name,
                    "engine": engine,
                    "config_id": config_id,
                    "seed": int(seed),
                    "severity_mode": severity_mode,
                    "calibration": c,
                    "fold_id": fold_assign,
                    "pred_freq": oof_freq_cal,
                    "pred_sev": oof_sev,
                    "pred_prime": oof_freq_cal * oof_sev,
                    "y_freq": y_freq_np,
                    "y_sev": y_sev_np,
                }
            )
        )
        pred_frames.append(
            pd.DataFrame(
                {
                    "row_idx": np.arange(len(X_test), dtype=int),
                    "is_test": 1,
                    "split": split_name,
                    "engine": engine,
                    "config_id": config_id,
                    "seed": int(seed),
                    "severity_mode": severity_mode,
                    "calibration": c,
                    "fold_id": np.nan,
                    "pred_freq": test_freq_cal,
                    "pred_sev": test_sev_mean,
                    "pred_prime": test_freq_cal * test_sev_mean,
                    "y_freq": np.nan,
                    "y_sev": np.nan,
                }
            )
        )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(run_rows),
        pd.concat(pred_frames, ignore_index=True),
    )

def _fit_predict_fold_v2(
    *,
    engine: str,
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
    e = engine.lower()
    if e == "catboost":
        return _fit_catboost_fold_v2(
            family=family,
            X_tr=X_tr,
            y_freq_tr=y_freq_tr,
            y_sev_tr=y_sev_tr,
            X_va=X_va,
            y_freq_va=y_freq_va,
            y_sev_va=y_sev_va,
            X_te=X_te,
            cat_cols=cat_cols,
            seed=seed,
            severity_mode=severity_mode,
            tweedie_power=tweedie_power,
            freq_params=freq_params,
            sev_params=sev_params,
            direct_params=direct_params,
        )
    if e == "lightgbm":
        return _fit_lgbm_fold_v2(
            family=family,
            X_tr=X_tr,
            y_freq_tr=y_freq_tr,
            y_sev_tr=y_sev_tr,
            X_va=X_va,
            y_freq_va=y_freq_va,
            y_sev_va=y_sev_va,
            X_te=X_te,
            cat_cols=cat_cols,
            seed=seed,
            severity_mode=severity_mode,
            tweedie_power=tweedie_power,
            freq_params=freq_params,
            sev_params=sev_params,
            direct_params=direct_params,
        )
    if e == "xgboost":
        return _fit_xgb_fold_v2(
            family=family,
            X_tr=X_tr,
            y_freq_tr=y_freq_tr,
            y_sev_tr=y_sev_tr,
            X_va=X_va,
            y_freq_va=y_freq_va,
            y_sev_va=y_sev_va,
            X_te=X_te,
            cat_cols=cat_cols,
            seed=seed,
            severity_mode=severity_mode,
            tweedie_power=tweedie_power,
            freq_params=freq_params,
            sev_params=sev_params,
            direct_params=direct_params,
        )
    raise ValueError(f"Unsupported engine: {engine}")

def _run_benchmark_single(
    spec: Mapping[str, Any],
    bundle: DatasetBundle,
    splits: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_set = str(spec.get("feature_set", "base_v2"))
    engine = str(spec.get("engine", "catboost")).lower()
    family = str(spec.get("family", "two_part_classic")).lower()
    config_id = str(spec.get("config_id", "v2_cfg"))
    severity_mode = str(spec.get("severity_mode", "classic")).lower()
    tweedie_power = float(spec.get("tweedie_power", 1.5))
    calibration_methods = list(spec.get("calibration_methods", ["none"]))
    if family == "direct_tweedie":
        calibration_methods = ["none"]
    use_tail_mapper = bool(spec.get("use_tail_mapper", False))
    use_target_encoding = bool(spec.get("use_target_encoding", False))
    target_encode_cols = list(spec.get("target_encode_cols", []))
    te_smoothing = float(spec.get("target_encoding_smoothing", 20.0))
    freq_params = dict(spec.get("freq_params", {}))
    sev_params = dict(spec.get("sev_params", {}))
    direct_params = dict(spec.get("direct_params", {}))
    split_names = list(spec.get("split_names", list(splits.keys())))

    X = bundle.X_train
    X_test = bundle.X_test
    y_freq = bundle.y_freq.to_numpy(dtype=int)
    y_sev = bundle.y_sev.to_numpy(dtype=float)
    base_cat_cols = list(bundle.cat_cols)
    n = len(X)
    n_test = len(X_test)

    all_fold_rows: List[pd.DataFrame] = []
    all_run_rows: List[pd.DataFrame] = []
    all_pred_rows: List[pd.DataFrame] = []

    for split_name in split_names:
        folds = splits[split_name]
        fold_assign = np.full(n, np.nan)
        oof_freq = np.full(n, np.nan)
        oof_sev = np.full(n, np.nan)
        oof_prime = np.full(n, np.nan)
        test_freq_parts: List[np.ndarray] = []
        test_sev_parts: List[np.ndarray] = []
        test_prime_parts: List[np.ndarray] = []
        fold_records: List[Dict[str, Any]] = []

        for fold_id, (tr_idx, va_idx) in folds.items():
            X_tr = X.iloc[tr_idx].copy()
            X_va = X.iloc[va_idx].copy()
            X_te = X_test.copy()
            y_freq_tr = y_freq[tr_idx]
            y_sev_tr = y_sev[tr_idx]
            y_freq_va = y_freq[va_idx]
            y_sev_va = y_sev[va_idx]

            if use_target_encoding and target_encode_cols:
                X_tr, X_va, X_te = _add_fold_target_encoding(
                    X_tr=X_tr,
                    y_freq_tr=y_freq_tr,
                    y_sev_tr=y_sev_tr,
                    X_va=X_va,
                    X_te=X_te,
                    cols=target_encode_cols,
                    smoothing=te_smoothing,
                )
            cat_cols_fold = [c for c in base_cat_cols if c in X_tr.columns and c in X_va.columns]

            p_va, m_va, prime_va, p_te, m_te, prime_te = _fit_predict_fold_v2(
                engine=engine,
                family=family,
                X_tr=X_tr,
                y_freq_tr=y_freq_tr,
                y_sev_tr=y_sev_tr,
                X_va=X_va,
                y_freq_va=y_freq_va,
                y_sev_va=y_sev_va,
                X_te=X_te,
                cat_cols=cat_cols_fold,
                seed=seed + int(fold_id),
                severity_mode=severity_mode,
                tweedie_power=tweedie_power,
                freq_params=freq_params,
                sev_params=sev_params,
                direct_params=direct_params,
            )
            p_va = np.clip(np.nan_to_num(p_va, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            p_te = np.clip(np.nan_to_num(p_te, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            m_va = np.maximum(np.nan_to_num(m_va, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            m_te = np.maximum(np.nan_to_num(m_te, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            prime_va = np.maximum(np.nan_to_num(prime_va, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            prime_te = np.maximum(np.nan_to_num(prime_te, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

            oof_freq[va_idx] = p_va
            oof_sev[va_idx] = m_va
            oof_prime[va_idx] = prime_va
            fold_assign[va_idx] = float(fold_id)
            test_freq_parts.append(p_te)
            test_sev_parts.append(m_te)
            test_prime_parts.append(prime_te)

            metrics = compute_metric_row(
                y_freq_true=y_freq_va,
                y_sev_true=y_sev_va,
                pred_freq=p_va,
                pred_sev=m_va,
                pred_prime=prime_va,
            )
            fold_records.append(
                {
                    "level": "fold",
                    "split": split_name,
                    "feature_set": feature_set,
                    "engine": engine,
                    "family": family,
                    "tweedie_power": float(tweedie_power),
                    "config_id": config_id,
                    "seed": int(seed),
                    "severity_mode": severity_mode,
                    "calibration": "none",
                    "tail_mapper": "none",
                    "fold_id": int(fold_id),
                    **metrics,
                }
            )

        valid = ~np.isnan(oof_prime)
        test_freq_mean = np.nanmean(np.vstack(test_freq_parts), axis=0)
        test_sev_mean = np.nanmean(np.vstack(test_sev_parts), axis=0)
        test_prime_mean = np.nanmean(np.vstack(test_prime_parts), axis=0)

        if use_tail_mapper and family in {"two_part_classic", "two_part_tweedie"}:
            oof_sev_tail = crossfit_tail_mapper_oof(
                pred_sev=oof_sev, y_sev=y_sev, y_freq=y_freq, fold_assign=fold_assign
            )
            pos_mask = valid & (y_freq == 1)
            full_mapper = fit_tail_mapper(oof_sev[pos_mask], y_sev[pos_mask])
            test_sev_tail = apply_tail_mapper(full_mapper, test_sev_mean)
            tail_mapper_name = str(full_mapper.get("kind", "identity"))
        else:
            oof_sev_tail = oof_sev.copy()
            test_sev_tail = test_sev_mean.copy()
            tail_mapper_name = "none"

        for calib in calibration_methods:
            c = str(calib).lower()
            if c == "none":
                oof_freq_cal = oof_freq.copy()
                test_freq_cal = test_freq_mean.copy()
            else:
                oof_freq_cal = crossfit_calibrate_oof(
                    probs=oof_freq, y_true=y_freq, fold_assign=fold_assign, method=c
                )
                full_cal = fit_calibrator(oof_freq[valid], y_freq[valid], c)
                test_freq_cal = apply_calibrator(full_cal, test_freq_mean, c)

            oof_freq_cal = np.clip(
                np.nan_to_num(oof_freq_cal, nan=float(np.nanmean(oof_freq[valid]))), 0.0, 1.0
            )
            test_freq_cal = np.clip(
                np.nan_to_num(test_freq_cal, nan=float(np.nanmean(oof_freq[valid]))), 0.0, 1.0
            )
            oof_sev_used = np.maximum(np.nan_to_num(oof_sev_tail, nan=0.0), 0.0)
            test_sev_used = np.maximum(np.nan_to_num(test_sev_tail, nan=0.0), 0.0)
            if family == "direct_tweedie":
                oof_prime_used = np.maximum(np.nan_to_num(oof_prime, nan=0.0), 0.0)
                test_prime_used = np.maximum(np.nan_to_num(test_prime_mean, nan=0.0), 0.0)
            else:
                oof_prime_used = np.maximum(oof_freq_cal * oof_sev_used, 0.0)
                test_prime_used = np.maximum(test_freq_cal * test_sev_used, 0.0)

            metrics = compute_metric_row(
                y_freq_true=y_freq[valid],
                y_sev_true=y_sev[valid],
                pred_freq=oof_freq_cal[valid],
                pred_sev=oof_sev_used[valid],
                pred_prime=oof_prime_used[valid],
            )
            all_run_rows.append(
                pd.DataFrame(
                    [
                        {
                            "level": "run",
                            "split": split_name,
                            "feature_set": feature_set,
                            "engine": engine,
                            "family": family,
                            "tweedie_power": float(tweedie_power),
                            "config_id": config_id,
                            "seed": int(seed),
                            "severity_mode": severity_mode,
                            "calibration": c,
                            "tail_mapper": tail_mapper_name,
                            "fold_id": -1,
                            "n_valid": int(valid.sum()),
                            **metrics,
                        }
                    ]
                )
            )
            all_pred_rows.append(
                pd.DataFrame(
                    {
                        "row_idx": np.arange(n, dtype=int),
                        "is_test": 0,
                        "split": split_name,
                        "feature_set": feature_set,
                        "engine": engine,
                        "family": family,
                        "tweedie_power": float(tweedie_power),
                        "config_id": config_id,
                        "seed": int(seed),
                        "severity_mode": severity_mode,
                        "calibration": c,
                        "tail_mapper": tail_mapper_name,
                        "fold_id": fold_assign,
                        "pred_freq": oof_freq_cal,
                        "pred_sev": oof_sev_used,
                        "pred_prime": oof_prime_used,
                        "y_freq": y_freq,
                        "y_sev": y_sev,
                    }
                )
            )
            all_pred_rows.append(
                pd.DataFrame(
                    {
                        "row_idx": np.arange(n_test, dtype=int),
                        "is_test": 1,
                        "split": split_name,
                        "feature_set": feature_set,
                        "engine": engine,
                        "family": family,
                        "tweedie_power": float(tweedie_power),
                        "config_id": config_id,
                        "seed": int(seed),
                        "severity_mode": severity_mode,
                        "calibration": c,
                        "tail_mapper": tail_mapper_name,
                        "fold_id": np.nan,
                        "pred_freq": test_freq_cal,
                        "pred_sev": test_sev_used,
                        "pred_prime": test_prime_used,
                        "y_freq": np.nan,
                        "y_sev": np.nan,
                    }
                )
            )
        all_fold_rows.append(pd.DataFrame(fold_records))

    fold_df = pd.concat(all_fold_rows, ignore_index=True) if all_fold_rows else pd.DataFrame()
    run_df = pd.concat(all_run_rows, ignore_index=True) if all_run_rows else pd.DataFrame()
    pred_df = pd.concat(all_pred_rows, ignore_index=True) if all_pred_rows else pd.DataFrame()
    if not fold_df.empty:
        fold_df["run_id"] = make_run_id(fold_df)
    if not run_df.empty:
        run_df["run_id"] = make_run_id(run_df)
    if not pred_df.empty:
        pred_df["run_id"] = make_run_id(pred_df)
    run_df, _ = _attach_run_health_columns(run_df, pred_df)
    return fold_df, run_df, pred_df

def run_benchmark(
    spec: Mapping[str, Any],
    bundle: DatasetBundle | Mapping[str, DatasetBundle],
    splits: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if isinstance(bundle, Mapping):
        fs_all = list(bundle.keys())
        fs_requested = list(spec.get("feature_sets", fs_all))
        fs_requested = [f for f in fs_requested if f in bundle]
        if not fs_requested:
            raise ValueError("No valid feature_set requested in spec['feature_sets'].")

        all_folds: List[pd.DataFrame] = []
        all_runs: List[pd.DataFrame] = []
        all_preds: List[pd.DataFrame] = []
        for fs in fs_requested:
            sp = dict(spec)
            sp["feature_set"] = fs
            f_df, r_df, p_df = _run_benchmark_single(sp, bundle=bundle[fs], splits=splits, seed=seed)
            all_folds.append(f_df)
            all_runs.append(r_df)
            all_preds.append(p_df)
        fold_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
        run_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
        pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
        return fold_df, run_df, pred_df

    return _run_benchmark_single(spec=spec, bundle=bundle, splits=splits, seed=seed)

