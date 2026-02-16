from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

TARGET_FREQ_COL = "nombre_sinistres"
TARGET_SEV_COL = "montant_sinistre"
INDEX_COL = "index"
ID_COLS = ["id_client", "id_vehicule", "id_contrat"]


@dataclass
class DatasetBundle:
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_freq: pd.Series
    y_sev: pd.Series
    feature_cols: List[str]
    cat_cols: List[str]
    num_cols: List[str]


class OrdinalFrameEncoder:
    def __init__(self, cat_cols: Sequence[str]):
        self.cat_cols = list(cat_cols)
        self.encoder: Optional[OrdinalEncoder] = None

    def fit(self, X: pd.DataFrame) -> "OrdinalFrameEncoder":
        if not self.cat_cols:
            return self
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float64,
        )
        self.encoder.fit(X[self.cat_cols].astype(str).fillna("NA"))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.encoder is not None and self.cat_cols:
            out.loc[:, self.cat_cols] = self.encoder.transform(
                out[self.cat_cols].astype(str).fillna("NA")
            )
        return out.astype(np.float64)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_train_test(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(data_dir)
    train = pd.read_csv(base / "train.csv")
    test = pd.read_csv(base / "test.csv")
    return train, test


def build_targets(train: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    y_sev = train[TARGET_SEV_COL].astype(float).rename("y_sev")
    y_freq = (y_sev > 0).astype(int).rename("y_freq")
    return y_freq, y_sev


def _binary_indicator_name(col: str, token: str) -> str:
    token = token.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return f"is_{col}_{token}"


def _add_binary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != "object":
            continue
        uniq = sorted({str(v) for v in out[c].dropna().unique() if str(v).strip()})
        if len(uniq) != 2:
            continue
        low = {v.lower() for v in uniq}
        if low == {"yes", "no"}:
            pos = "yes"
        elif low == {"true", "false"}:
            pos = "true"
        elif low == {"m", "f"}:
            pos = "m"
        else:
            pos = uniq[0]
        out[_binary_indicator_name(c, pos)] = (
            out[c].astype(str).str.lower().eq(str(pos).lower()).astype(int)
        )
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "conducteur2" in out.columns:
        second_driver = ~out["conducteur2"].astype(str).str.lower().eq("no")
        out["has_second_driver"] = second_driver.astype(int)
        for c in ["age_conducteur2", "anciennete_permis2"]:
            if c in out.columns:
                out.loc[~second_driver, c] = np.nan

    for c in ["poids_vehicule", "cylindre_vehicule"]:
        if c in out.columns:
            out.loc[out[c] == 0, c] = np.nan

    if "code_postal" in out.columns:
        cp = out["code_postal"].astype(str).str.zfill(5)
        out["code_postal"] = cp
        out["cp2"] = cp.str[:2]
        out["cp3"] = cp.str[:3]

    if {"prix_vehicule", "poids_vehicule"}.issubset(out.columns):
        out["prix_par_kg"] = out["prix_vehicule"] / out["poids_vehicule"].replace(0, np.nan)

    if {"din_vehicule", "cylindre_vehicule"}.issubset(out.columns):
        out["din_par_cylindre"] = (
            out["din_vehicule"] / out["cylindre_vehicule"].replace(0, np.nan)
        )

    out = _add_binary_indicators(out)

    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).fillna("NA")

    return out


def prepare_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    drop_identifiers: bool = True,
) -> DatasetBundle:
    train_fe = add_engineered_features(train)
    test_fe = add_engineered_features(test)
    y_freq, y_sev = build_targets(train_fe)

    exclude = {INDEX_COL, TARGET_FREQ_COL, TARGET_SEV_COL}
    cols = [c for c in train_fe.columns if c in test_fe.columns and c not in exclude]
    if drop_identifiers:
        cols = [c for c in cols if c not in ID_COLS]

    X_train = train_fe[cols].copy()
    X_test = test_fe[cols].copy()
    cat_cols = [c for c in cols if X_train[c].dtype == "object"]
    num_cols = [c for c in cols if c not in cat_cols]

    return DatasetBundle(
        train_raw=train,
        test_raw=test,
        X_train=X_train,
        X_test=X_test,
        y_freq=y_freq,
        y_sev=y_sev,
        feature_cols=cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )


def build_primary_time_folds(
    train: pd.DataFrame,
    *,
    n_blocks: int = 5,
    index_col: str = INDEX_COL,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(train[index_col].to_numpy())
    blocks = np.array_split(order, n_blocks)
    folds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for fold in range(1, n_blocks):
        tr = np.concatenate(blocks[:fold]).astype(int)
        va = blocks[fold].astype(int)
        folds[fold] = (tr, va)
    return folds


def build_secondary_group_folds(
    train: pd.DataFrame,
    *,
    n_splits: int = 5,
    group_col: str = "id_client",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    groups = train[group_col].to_numpy()
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for i, (tr, va) in enumerate(gkf.split(train, groups=groups), start=1):
        out[i] = (tr.astype(int), va.astype(int))
    return out


def validate_folds_disjoint(
    folds: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    *,
    check_full_coverage: bool = False,
    n_rows: Optional[int] = None,
) -> None:
    valid_union: List[int] = []
    for fold, (tr, va) in folds.items():
        inter = set(map(int, tr)).intersection(set(map(int, va)))
        if inter:
            raise AssertionError(f"Fold {fold} has train/valid overlap")
        valid_union.extend(list(map(int, va)))
    if check_full_coverage:
        if n_rows is None:
            raise ValueError("n_rows is required when check_full_coverage=True")
        expected = set(range(n_rows))
        got = set(valid_union)
        if expected != got:
            raise AssertionError(f"Coverage mismatch. Missing={len(expected - got)}")


def folds_to_frame(
    folds: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    *,
    split_name: str,
    n_rows: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold, (tr, va) in folds.items():
        rows.extend(
            {"split": split_name, "fold_id": int(fold), "row_idx": int(i), "role": "train"}
            for i in tr
        )
        rows.extend(
            {"split": split_name, "fold_id": int(fold), "row_idx": int(i), "role": "valid"}
            for i in va
        )
    df = pd.DataFrame(rows)
    df["n_rows_total"] = int(n_rows)
    return df


def export_fold_artifacts(
    *,
    train: pd.DataFrame,
    primary_folds: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    secondary_folds: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path = "artifacts",
) -> None:
    out = ensure_dir(output_dir)
    folds_to_frame(
        primary_folds, split_name="primary_time", n_rows=len(train)
    ).to_parquet(out / "folds_primary.parquet", index=False)
    folds_to_frame(
        secondary_folds, split_name="secondary_group", n_rows=len(train)
    ).to_parquet(out / "folds_secondary.parquet", index=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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
) -> Dict[str, float]:
    pred_freq = np.nan_to_num(np.asarray(pred_freq, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    pred_sev = np.nan_to_num(np.asarray(pred_sev, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_freq_true = np.asarray(y_freq_true, dtype=int)
    y_sev_true = np.asarray(y_sev_true, dtype=float)

    pred_prime = pred_freq * pred_sev
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


def _severity_fallback(y_sev_tr: np.ndarray, n_va: int, n_te: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(y_sev_tr, dtype=float)
    pos = pos[pos > 0]
    base = float(np.nanmean(pos)) if len(pos) else 0.0
    return np.full(n_va, base, dtype=float), np.full(n_te, base, dtype=float)


def fit_calibrator(probs: np.ndarray, y_true: np.ndarray, method: str):
    m = method.lower()
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if m == "none":
        return None
    if m == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(p, y)
        return model
    if m == "platt":
        model = LogisticRegression(max_iter=2000)
        model.fit(p.reshape(-1, 1), y)
        return model
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(model, probs: np.ndarray, method: str) -> np.ndarray:
    m = method.lower()
    p = np.asarray(probs, dtype=float)
    if m == "none" or model is None:
        return p
    if m == "isotonic":
        return model.transform(p)
    if m == "platt":
        return model.predict_proba(p.reshape(-1, 1))[:, 1]
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


def _fit_lgbm(
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
    from lightgbm import LGBMClassifier, LGBMRegressor

    enc = OrdinalFrameEncoder(cat_cols).fit(X_tr)
    Xtr = enc.transform(X_tr)
    Xva = enc.transform(X_va)
    Xte = enc.transform(X_te)

    f_params = {
        "objective": "binary",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }
    f_params.update(freq_params)

    s_params = {
        "objective": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
    }
    s_params.update(sev_params)

    clf = LGBMClassifier(**f_params)
    clf.fit(Xtr, y_freq_tr)
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    pos = y_freq_tr == 1
    if int(pos.sum()) < 5:
        m_va, m_te = _severity_fallback(y_sev_tr, len(X_va), len(X_te))
        return p_va, m_va, p_te, m_te

    y_pos = y_sev_tr[pos]
    y_log = np.log1p(y_pos)
    w = make_tail_weights(y_pos) if severity_mode == "weighted_tail" else None
    reg = LGBMRegressor(**s_params)
    reg.fit(Xtr.loc[pos], y_log, sample_weight=w)
    z_va = reg.predict(Xva)
    z_te = reg.predict(Xte)
    z_tr = reg.predict(Xtr.loc[pos])
    resid = y_log - z_tr
    smear = float(np.average(np.exp(resid), weights=w)) if w is not None else float(np.mean(np.exp(resid)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    m_va = np.maximum(smear * np.exp(z_va) - 1.0, 0.0)
    m_te = np.maximum(smear * np.exp(z_te) - 1.0, 0.0)
    m_va = np.nan_to_num(m_va, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
    m_te = np.nan_to_num(m_te, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
    return p_va, m_va, p_te, m_te


def _fit_xgb(
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
    smear = float(np.average(np.exp(resid), weights=w)) if w is not None else float(np.mean(np.exp(resid)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    m_va = np.maximum(smear * np.exp(z_va) - 1.0, 0.0)
    m_te = np.maximum(smear * np.exp(z_te) - 1.0, 0.0)
    m_va = np.nan_to_num(m_va, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
    m_te = np.nan_to_num(m_te, nan=float(np.nanmean(y_pos) if len(y_pos) else 0.0), posinf=0.0, neginf=0.0)
    return p_va, m_va, p_te, m_te


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


def optimize_non_negative_weights(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n_models = p.shape[1]
    x0 = np.full(n_models, 1.0 / n_models)
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        return rmse(y, p @ w)

    r = minimize(objective, x0=x0, bounds=bounds, constraints=constraints)
    if not r.success:
        return x0
    w = np.clip(r.x, 0.0, 1.0)
    s = w.sum()
    return x0 if s <= 0 else w / s


def simulate_public_private_shakeup(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    n_sim: int = 2000,
    public_ratio: float = 1.0 / 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    n_pub = int(round(n * public_ratio))
    idx = np.arange(n)
    rows: List[Dict[str, float]] = []
    for s in range(n_sim):
        rng.shuffle(idx)
        pub = idx[:n_pub]
        pri = idx[n_pub:]
        pub_rmse = rmse(y[pub], p[pub])
        pri_rmse = rmse(y[pri], p[pri])
        rows.append(
            {
                "sim_id": s,
                "rmse_public": pub_rmse,
                "rmse_private": pri_rmse,
                "gap_public_minus_private": pub_rmse - pri_rmse,
            }
        )
    return pd.DataFrame(rows)


def build_submission(index_series: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    sub = pd.DataFrame({"index": index_series.astype(int).to_numpy(), "pred": np.asarray(pred, dtype=float)})
    sub["pred"] = sub["pred"].clip(lower=0.0)
    return sub


COARSE_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "catboost": [
        {
            "config_id": "cb_c1",
            "freq_params": {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
            "sev_params": {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
        },
        {
            "config_id": "cb_c2",
            "freq_params": {"depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
            "sev_params": {"depth": 9, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
        },
        {
            "config_id": "cb_c3",
            "freq_params": {"depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
            "sev_params": {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
        },
    ],
    "lightgbm": [
        {
            "config_id": "lgb_c1",
            "freq_params": {"num_leaves": 63, "learning_rate": 0.03},
            "sev_params": {"num_leaves": 127, "learning_rate": 0.03},
        },
        {
            "config_id": "lgb_c2",
            "freq_params": {"num_leaves": 95, "learning_rate": 0.02, "min_child_samples": 80},
            "sev_params": {"num_leaves": 159, "learning_rate": 0.02, "min_child_samples": 60},
        },
        {
            "config_id": "lgb_c3",
            "freq_params": {"num_leaves": 47, "learning_rate": 0.04, "min_child_samples": 40},
            "sev_params": {"num_leaves": 95, "learning_rate": 0.04, "min_child_samples": 40},
        },
    ],
    "xgboost": [
        {
            "config_id": "xgb_c1",
            "freq_params": {"max_depth": 6, "learning_rate": 0.03},
            "sev_params": {"max_depth": 8, "learning_rate": 0.03},
        },
        {
            "config_id": "xgb_c2",
            "freq_params": {"max_depth": 5, "learning_rate": 0.04, "min_child_weight": 8},
            "sev_params": {"max_depth": 7, "learning_rate": 0.04, "min_child_weight": 8},
        },
        {
            "config_id": "xgb_c3",
            "freq_params": {"max_depth": 7, "learning_rate": 0.02, "min_child_weight": 4},
            "sev_params": {"max_depth": 9, "learning_rate": 0.02, "min_child_weight": 4},
        },
    ],
}


def pick_top_configs(
    run_registry: pd.DataFrame,
    *,
    split_name: str = "primary_time",
    top_k_per_engine: int = 2,
) -> Dict[str, List[str]]:
    rr = run_registry.copy()
    rr = rr[(rr["level"] == "run") & (rr["split"] == split_name)]
    rr = rr.sort_values(["engine", "rmse_prime", "brier_freq"])
    out: Dict[str, List[str]] = {}
    for engine, g in rr.groupby("engine"):
        out[engine] = g["config_id"].drop_duplicates().head(top_k_per_engine).tolist()
    return out
