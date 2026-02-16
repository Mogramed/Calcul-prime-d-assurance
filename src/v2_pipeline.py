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
DEFAULT_V2_DIR = Path("artifacts") / "v2"


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


# ---------------------------
# V2 extension layer
# ---------------------------


def _normalize_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].fillna("NA").astype(str)
    return out


def _build_rare_maps(
    train_df: pd.DataFrame, *, cat_cols: Sequence[str], min_count: int = 30
) -> Dict[str, set]:
    maps: Dict[str, set] = {}
    for c in cat_cols:
        if c not in train_df.columns:
            continue
        vc = train_df[c].astype(str).value_counts(dropna=False)
        maps[c] = set(vc[vc < min_count].index.astype(str))
    return maps


def _apply_rare_maps(df: pd.DataFrame, rare_maps: Mapping[str, set]) -> pd.DataFrame:
    out = df.copy()
    for c, rare in rare_maps.items():
        if c not in out.columns:
            continue
        s = out[c].astype(str)
        out[c] = np.where(s.isin(rare), "__RARE__", s)
    return out


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    out = np.divide(aa, bb, out=np.full_like(aa, np.nan), where=np.isfinite(bb) & (bb != 0))
    return out


def _add_engineered_features_core_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = add_engineered_features(df)

    if {"marque_vehicule", "modele_vehicule"}.issubset(out.columns):
        out["marque_modele"] = (
            out["marque_vehicule"].astype(str) + "__" + out["modele_vehicule"].astype(str)
        )

    if {"din_vehicule", "poids_vehicule"}.issubset(out.columns):
        out["power_weight_ratio"] = _safe_div(out["din_vehicule"], out["poids_vehicule"])

    if "anciennete_vehicule" in out.columns:
        bins = [-np.inf, 2, 5, 9, 14, np.inf]
        labels = ["v_new", "v_recent", "v_mid", "v_old", "v_very_old"]
        out["veh_age_band"] = pd.cut(out["anciennete_vehicule"], bins=bins, labels=labels).astype(str)

    if "power_weight_ratio" in out.columns:
        bins = [-np.inf, 0.04, 0.06, 0.08, 0.12, np.inf]
        labels = ["pw_low", "pw_mid", "pw_high", "pw_perf", "pw_extreme"]
        out["power_weight_band"] = pd.cut(out["power_weight_ratio"], bins=bins, labels=labels).astype(str)

    if {"bonus", "utilisation"}.issubset(out.columns):
        out["bonus_x_usage"] = out["bonus"].astype(float) * (
            out["utilisation"].astype(str).str.len().astype(float)
        )

    if {"age_conducteur1", "anciennete_permis1"}.issubset(out.columns):
        out["age_x_permis"] = out["age_conducteur1"].astype(float) * out[
            "anciennete_permis1"
        ].astype(float)

    if {"prix_vehicule", "type_vehicule"}.issubset(out.columns):
        out["prix_x_type_vehicule"] = (
            out["type_vehicule"].astype(str)
            + "__"
            + pd.cut(
                out["prix_vehicule"].astype(float),
                bins=[-np.inf, 10000, 20000, 35000, np.inf],
                labels=["p_low", "p_mid", "p_hi", "p_lux"],
            ).astype(str)
        )

    out = _normalize_object_cols(out)
    return out


def add_engineered_features_v2(
    train: pd.DataFrame, test: pd.DataFrame, *, rare_min_count: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = _add_engineered_features_core_v2(train)
    te = _add_engineered_features_core_v2(test)

    cat_cols = [c for c in tr.columns if tr[c].dtype == "object" and c in te.columns]
    rare_maps = _build_rare_maps(tr, cat_cols=cat_cols, min_count=rare_min_count)
    tr = _apply_rare_maps(tr, rare_maps)
    te = _apply_rare_maps(te, rare_maps)

    if "code_postal" in tr.columns and "code_postal" in te.columns:
        seen_cp = set(tr["code_postal"].astype(str).unique())
        tr["is_unseen_cp5"] = 0
        te["is_unseen_cp5"] = (~te["code_postal"].astype(str).isin(seen_cp)).astype(int)
    else:
        tr["is_unseen_cp5"] = 0
        te["is_unseen_cp5"] = 0

    if "modele_vehicule" in tr.columns and "modele_vehicule" in te.columns:
        seen_modele = set(tr["modele_vehicule"].astype(str).unique())
        tr["is_unseen_modele"] = 0
        te["is_unseen_modele"] = (~te["modele_vehicule"].astype(str).isin(seen_modele)).astype(int)
    else:
        tr["is_unseen_modele"] = 0
        te["is_unseen_modele"] = 0

    if "marque_modele" in tr.columns:
        vc = tr["marque_modele"].astype(str).value_counts(dropna=False)
        rare_mm = set(vc[vc < 20].index.astype(str))
        seen_mm = set(vc.index.astype(str))
        tr["is_rare_marque_modele"] = tr["marque_modele"].astype(str).isin(rare_mm).astype(int)
        te_mm = te["marque_modele"].astype(str) if "marque_modele" in te.columns else pd.Series("", index=te.index)
        te["is_rare_marque_modele"] = (
            te_mm.isin(rare_mm) | (~te_mm.isin(seen_mm))
        ).astype(int)
    else:
        tr["is_rare_marque_modele"] = 0
        te["is_rare_marque_modele"] = 0
    return tr, te


def _make_bundle(train_fe: pd.DataFrame, test_fe: pd.DataFrame, *, keep_cols: Sequence[str], name: str) -> DatasetBundle:
    y_freq, y_sev = build_targets(train_fe)
    cols = [c for c in keep_cols if c in train_fe.columns and c in test_fe.columns]
    X_train = train_fe[cols].copy()
    X_test = test_fe[cols].copy()
    cat_cols = [c for c in cols if X_train[c].dtype == "object"]
    num_cols = [c for c in cols if c not in cat_cols]
    return DatasetBundle(
        train_raw=train_fe,
        test_raw=test_fe,
        X_train=X_train,
        X_test=X_test,
        y_freq=y_freq,
        y_sev=y_sev,
        feature_cols=cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )


def prepare_feature_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    rare_min_count: int = 30,
    drop_identifiers: bool = True,
) -> Dict[str, DatasetBundle]:
    train_fe, test_fe = add_engineered_features_v2(train, test, rare_min_count=rare_min_count)
    exclude = {INDEX_COL, TARGET_FREQ_COL, TARGET_SEV_COL}
    all_cols = [c for c in train_fe.columns if c in test_fe.columns and c not in exclude]
    if drop_identifiers:
        all_cols = [c for c in all_cols if c not in ID_COLS]

    robust_drop = {"code_postal", "modele_vehicule", "marque_modele"}
    compact_drop = robust_drop.union({"marque_vehicule"})

    return {
        "base_v2": _make_bundle(train_fe, test_fe, keep_cols=all_cols, name="base_v2"),
        "robust_v2": _make_bundle(
            train_fe, test_fe, keep_cols=[c for c in all_cols if c not in robust_drop], name="robust_v2"
        ),
        "compact_v2": _make_bundle(
            train_fe, test_fe, keep_cols=[c for c in all_cols if c not in compact_drop], name="compact_v2"
        ),
    }


def build_aux_blocked_folds(
    train: pd.DataFrame, *, n_blocks: int = 5, index_col: str = INDEX_COL
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    order = np.argsort(train[index_col].to_numpy())
    blocks = np.array_split(order, n_blocks)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for i, va in enumerate(blocks, start=1):
        tr = np.concatenate([b for j, b in enumerate(blocks) if j != (i - 1)]).astype(int)
        out[i] = (tr, va.astype(int))
    return out


def build_split_registry(
    train: pd.DataFrame,
    *,
    n_blocks_time: int = 5,
    n_splits_group: int = 5,
    group_col: str = "id_client",
) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    return {
        "primary_time": build_primary_time_folds(train, n_blocks=n_blocks_time),
        "secondary_group": build_secondary_group_folds(train, n_splits=n_splits_group, group_col=group_col),
        "aux_blocked5": build_aux_blocked_folds(train, n_blocks=5),
    }


def validate_group_disjoint(
    folds: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    groups: pd.Series | np.ndarray,
) -> None:
    g = pd.Series(groups).astype(str).to_numpy()
    for fold_id, (tr, va) in folds.items():
        if set(g[tr]).intersection(set(g[va])):
            raise AssertionError(f"Group overlap in fold {fold_id}")


def export_split_artifacts_v2(
    *,
    train: pd.DataFrame,
    splits: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    output_dir: str | Path = DEFAULT_V2_DIR,
) -> None:
    out = ensure_dir(output_dir)
    names = {
        "primary_time": "folds_primary_time.parquet",
        "secondary_group": "folds_secondary_group.parquet",
        "aux_blocked5": "folds_aux_blocked5.parquet",
    }
    for split_name, folds in splits.items():
        if split_name not in names:
            continue
        folds_to_frame(folds, split_name=split_name, n_rows=len(train)).to_parquet(
            out / names[split_name], index=False
        )


def fit_tail_mapper(
    oof_pred_sev_pos: np.ndarray,
    y_pos: np.ndarray,
    *,
    min_samples: int = 150,
) -> Dict[str, Any]:
    x = np.asarray(oof_pred_sev_pos, dtype=float)
    y = np.asarray(y_pos, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = x[mask]
    y = y[mask]
    if len(x) < min_samples or len(np.unique(x)) < 10:
        return {"kind": "identity"}
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    return {
        "kind": "isotonic",
        "x_thresholds": iso.X_thresholds_.astype(float).tolist(),
        "y_thresholds": iso.y_thresholds_.astype(float).tolist(),
    }


def apply_tail_mapper(mapper: Mapping[str, Any], pred_sev: np.ndarray) -> np.ndarray:
    p = np.asarray(pred_sev, dtype=float)
    kind = str(mapper.get("kind", "identity")).lower()
    if kind == "identity":
        return np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if kind == "isotonic":
        x = np.asarray(mapper.get("x_thresholds", []), dtype=float)
        y = np.asarray(mapper.get("y_thresholds", []), dtype=float)
        if len(x) < 2 or len(y) < 2:
            return np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        mapped = np.interp(p, x, y, left=y[0], right=y[-1])
        return np.maximum(np.nan_to_num(mapped, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    raise ValueError(f"Unknown mapper kind: {kind}")


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


def make_run_id(df: pd.DataFrame) -> pd.Series:
    if "run_id" in df.columns:
        return df["run_id"].astype(str)

    cols_new = [
        "feature_set",
        "engine",
        "family",
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

    cols_v1 = ["engine", "config_id", "seed", "severity_mode", "calibration"]
    if all(c in df.columns for c in cols_v1):
        out = df[cols_v1[0]].astype(str)
        for c in cols_v1[1:]:
            out = out + "|" + df[c].astype(str)
        return out

    raise KeyError("Cannot build run_id: required columns are missing.")


def compute_ood_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    cat_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if cat_cols is None:
        cat_cols = [c for c in train.columns if train[c].dtype == "object" and c in test.columns]
    rows: List[Dict[str, Any]] = []
    for c in cat_cols:
        tr_u = set(train[c].astype(str).dropna().unique())
        te_u = set(test[c].astype(str).dropna().unique())
        unseen = te_u - tr_u
        rows.append(
            {
                "diagnostic_type": "ood",
                "feature": c,
                "train_unique": int(len(tr_u)),
                "test_unique": int(len(te_u)),
                "unseen_test_levels": int(len(unseen)),
                "unseen_ratio_on_levels": float(len(unseen) / max(len(te_u), 1)),
            }
        )
    return pd.DataFrame(rows)


def compute_segment_bias_from_oof(
    train: pd.DataFrame,
    oof_df: pd.DataFrame,
    *,
    run_id: str,
    split_name: str = "primary_time",
    segment_cols: Optional[Sequence[str]] = None,
    min_count: int = 150,
) -> pd.DataFrame:
    if segment_cols is None:
        segment_cols = [
            "utilisation",
            "type_contrat",
            "cp2",
            "cp3",
            "marque_vehicule",
            "modele_vehicule",
        ]
    d = oof_df.copy()
    if "run_id" not in d.columns:
        d["run_id"] = make_run_id(d)
    d = d[(d["is_test"] == 0) & (d["split"] == split_name) & (d["run_id"] == run_id)].copy()
    d = d.sort_values("row_idx")
    valid = d["pred_prime"].notna().to_numpy()
    tr = train.reset_index(drop=True).loc[valid].copy()
    dd = d.loc[valid].copy().reset_index(drop=True)
    tr["y_true"] = dd["y_sev"].to_numpy()
    tr["pred_prime"] = dd["pred_prime"].to_numpy()
    tr["error"] = tr["pred_prime"] - tr["y_true"]
    rows: List[Dict[str, Any]] = []
    for c in segment_cols:
        if c not in tr.columns:
            continue
        grp = (
            tr.groupby(c)
            .agg(
                n=("y_true", "size"),
                y_mean=("y_true", "mean"),
                p_mean=("pred_prime", "mean"),
                bias=("error", "mean"),
            )
            .reset_index()
        )
        grp = grp[grp["n"] >= min_count]
        for _, r in grp.iterrows():
            rows.append(
                {
                    "diagnostic_type": "segment_bias",
                    "feature": c,
                    "segment": str(r[c]),
                    "n": int(r["n"]),
                    "y_mean": float(r["y_mean"]),
                    "p_mean": float(r["p_mean"]),
                    "bias": float(r["bias"]),
                }
            )
    return pd.DataFrame(rows)


def build_model_cards(run_registry: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rr = run_registry.copy()
    if "run_id" not in rr.columns:
        rr["run_id"] = make_run_id(rr)
    sel = selected[["run_id"]].drop_duplicates().copy()
    card = rr[rr["run_id"].isin(set(sel["run_id"]))].copy()
    cols = [
        "run_id",
        "feature_set",
        "engine",
        "family",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
        "split",
        "rmse_prime",
        "auc_freq",
        "brier_freq",
        "rmse_sev_pos",
        "q99_ratio_pos",
    ]
    cols = [c for c in cols if c in card.columns]
    return card[cols].sort_values(["run_id", "split"]).reset_index(drop=True)


def simulate_public_private_shakeup_v2(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    n_sim: int = 2000,
    public_ratio: float = 1.0 / 3.0,
    seed: int = 42,
    stratified_tail: bool = False,
    tail_quantile: float = 0.9,
    tail_public_share: float = 0.5,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    n_pub = int(round(n * public_ratio))
    idx = np.arange(n)
    rows: List[Dict[str, float]] = []

    if stratified_tail:
        thr = float(np.nanquantile(y, tail_quantile))
        tail_idx = idx[y >= thr]
        body_idx = idx[y < thr]

    for s in range(n_sim):
        if stratified_tail:
            rng.shuffle(tail_idx)
            rng.shuffle(body_idx)
            n_tail_pub = min(len(tail_idx), int(round(n_pub * tail_public_share)))
            n_body_pub = n_pub - n_tail_pub
            pub = np.concatenate([tail_idx[:n_tail_pub], body_idx[:n_body_pub]])
            pri = np.setdiff1d(idx, pub, assume_unique=False)
        else:
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
                "stratified_tail": int(stratified_tail),
            }
        )
    return pd.DataFrame(rows)


def _smooth_target_encoding_map(
    x: pd.Series,
    y: pd.Series,
    *,
    smoothing: float,
) -> Tuple[Dict[str, float], float]:
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _fit_lgbm_fold_v2(
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
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping

    enc = OrdinalFrameEncoder(cat_cols).fit(X_tr)
    Xtr = enc.transform(X_tr)
    Xva = enc.transform(X_va)
    Xte = enc.transform(X_te)
    callbacks = [early_stopping(stopping_rounds=100, verbose=False)]

    f_params = {
        "objective": "binary",
        "n_estimators": 6000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    f_params.update(freq_params)
    clf = LGBMClassifier(**f_params)
    clf.fit(
        Xtr,
        y_freq_tr,
        eval_set=[(Xva, y_freq_va)],
        eval_metric="binary_logloss",
        callbacks=callbacks,
    )
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    fam = family.lower()
    sev_mode = severity_mode.lower()
    if fam == "direct_tweedie":
        y_target = np.clip(y_sev_tr.astype(float), 0.0, None)
        if sev_mode == "winsorized":
            y_target = _apply_winsor(y_target, quantile=0.995)
        d_params = {
            "objective": "tweedie",
            "tweedie_variance_power": float(tweedie_power),
            "n_estimators": 7000,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        d_params.update(direct_params)
        reg = LGBMRegressor(**d_params)
        reg.fit(
            Xtr,
            y_target,
            eval_set=[(Xva, np.clip(y_sev_va.astype(float), 0.0, None))],
            eval_metric="rmse",
            callbacks=callbacks,
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
            "objective": "tweedie",
            "tweedie_variance_power": float(tweedie_power),
            "n_estimators": 7000,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        s_params.update(sev_params)
        reg = LGBMRegressor(**s_params)
        reg.fit(
            Xtr.loc[pos],
            y_pos_fit,
            sample_weight=w,
            eval_set=[(Xva, np.clip(y_sev_va.astype(float), 0.0, None))],
            eval_metric="rmse",
            callbacks=callbacks,
        )
        m_va = np.maximum(reg.predict(Xva), 0.0)
        m_te = np.maximum(reg.predict(Xte), 0.0)
    else:
        s_params = {
            "objective": "rmse",
            "n_estimators": 7000,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        s_params.update(sev_params)
        reg = LGBMRegressor(**s_params)
        y_log = np.log1p(y_pos_fit)
        y_va_log = np.log1p(np.clip(y_sev_va.astype(float), 0.0, None))
        reg.fit(
            Xtr.loc[pos],
            y_log,
            sample_weight=w,
            eval_set=[(Xva, y_va_log)],
            eval_metric="rmse",
            callbacks=callbacks,
        )
        z_va = reg.predict(Xva)
        z_te = reg.predict(Xte)
        z_tr = reg.predict(Xtr.loc[pos])
        m_va, m_te = _smearing_inverse(y_pos_fit, z_tr=z_tr, z_va=z_va, z_te=z_te, sample_weight=w)

    prime_va = np.maximum(p_va * m_va, 0.0)
    prime_te = np.maximum(p_te * m_te, 0.0)
    return p_va, m_va, prime_va, p_te, m_te, prime_te


def _fit_xgb_fold_v2(
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


def run_benchmark(
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
    return fold_df, run_df, pred_df


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


def select_final_models(
    run_registry: pd.DataFrame,
    risk_policy: str | Mapping[str, Any] = "stability_private",
) -> pd.DataFrame:
    if isinstance(risk_policy, str):
        rp = risk_policy.lower()
        if rp == "stability_private":
            policy = {
                "max_secondary_gap": 1.5,
                "max_aux_gap": 1.5,
                "max_models": 6,
                "min_q99_ratio": 0.25,
                "tail_penalty_weight": 3.0,
            }
        elif rp == "balanced":
            policy = {
                "max_secondary_gap": 3.0,
                "max_aux_gap": 3.0,
                "max_models": 6,
                "min_q99_ratio": 0.20,
                "tail_penalty_weight": 2.0,
            }
        else:
            policy = {
                "max_secondary_gap": 6.0,
                "max_aux_gap": 6.0,
                "max_models": 6,
                "min_q99_ratio": 0.10,
                "tail_penalty_weight": 1.0,
            }
    else:
        policy = dict(risk_policy)

    rr = run_registry.copy()
    if "run_id" not in rr.columns:
        rr["run_id"] = make_run_id(rr)
    rr = rr[rr["level"] == "run"].copy()

    piv_rmse = rr.pivot_table(index="run_id", columns="split", values="rmse_prime", aggfunc="mean")
    piv_q99 = rr.pivot_table(index="run_id", columns="split", values="q99_ratio_pos", aggfunc="mean")
    meta_cols = [
        "feature_set",
        "engine",
        "family",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
    ]
    meta = rr.groupby("run_id")[meta_cols].first()
    out = meta.join(piv_rmse.add_prefix("rmse_"), how="left").join(
        piv_q99.add_prefix("q99_"), how="left"
    )
    out = out.reset_index()
    for c in ["rmse_primary_time", "rmse_secondary_group", "rmse_aux_blocked5", "q99_primary_time"]:
        if c not in out.columns:
            out[c] = np.nan
    out["gap_secondary"] = out["rmse_secondary_group"] - out["rmse_primary_time"]
    out["gap_aux"] = out["rmse_aux_blocked5"] - out["rmse_primary_time"]
    out["tail_penalty"] = (1.0 - out["q99_primary_time"]).abs()
    out["accepted"] = (
        (out["gap_secondary"].fillna(0.0) <= float(policy["max_secondary_gap"]))
        & (out["gap_aux"].fillna(0.0) <= float(policy["max_aux_gap"]))
        & (out["q99_primary_time"].fillna(0.0) >= float(policy["min_q99_ratio"]))
    )
    out["selection_score"] = (
        out["rmse_primary_time"].fillna(np.inf)
        + np.maximum(out["gap_secondary"].fillna(0.0), 0.0)
        + np.maximum(out["gap_aux"].fillna(0.0), 0.0)
        + float(policy["tail_penalty_weight"]) * out["tail_penalty"].fillna(1.0)
    )
    out = out.sort_values(["accepted", "selection_score"], ascending=[False, True]).reset_index(drop=True)
    sel = out[out["accepted"]].head(int(policy["max_models"]))
    if sel.empty:
        sel = out.head(int(policy["max_models"]))
    return sel.reset_index(drop=True)


V2_COARSE_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "catboost": [
        {
            "config_id": "cb_v2_c1",
            "freq_params": {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
            "sev_params": {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
            "direct_params": {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 8.0},
        },
        {
            "config_id": "cb_v2_c2",
            "freq_params": {"depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
            "sev_params": {"depth": 9, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
            "direct_params": {"depth": 9, "learning_rate": 0.025, "l2_leaf_reg": 10.0},
        },
        {
            "config_id": "cb_v2_c3",
            "freq_params": {"depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
            "sev_params": {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
            "direct_params": {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 6.0},
        },
        {
            "config_id": "cb_v2_c4",
            "freq_params": {"depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 12.0},
            "sev_params": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 12.0},
            "direct_params": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 12.0},
        },
        {
            "config_id": "cb_v2_c5",
            "freq_params": {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 9.0},
            "sev_params": {"depth": 10, "learning_rate": 0.02, "l2_leaf_reg": 9.0},
            "direct_params": {"depth": 10, "learning_rate": 0.02, "l2_leaf_reg": 9.0},
        },
        {
            "config_id": "cb_v2_c6",
            "freq_params": {"depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
            "sev_params": {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
            "direct_params": {"depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 5.0},
        },
    ],
    "lightgbm": [
        {
            "config_id": "lgb_v2_c1",
            "freq_params": {"num_leaves": 63, "learning_rate": 0.03},
            "sev_params": {"num_leaves": 127, "learning_rate": 0.03},
            "direct_params": {"num_leaves": 127, "learning_rate": 0.03},
        },
        {
            "config_id": "lgb_v2_c2",
            "freq_params": {"num_leaves": 95, "learning_rate": 0.02, "min_child_samples": 80},
            "sev_params": {"num_leaves": 159, "learning_rate": 0.02, "min_child_samples": 60},
            "direct_params": {"num_leaves": 159, "learning_rate": 0.02, "min_child_samples": 60},
        },
        {
            "config_id": "lgb_v2_c3",
            "freq_params": {"num_leaves": 47, "learning_rate": 0.04, "min_child_samples": 40},
            "sev_params": {"num_leaves": 95, "learning_rate": 0.04, "min_child_samples": 40},
            "direct_params": {"num_leaves": 95, "learning_rate": 0.04, "min_child_samples": 40},
        },
        {
            "config_id": "lgb_v2_c4",
            "freq_params": {"num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 120},
            "sev_params": {"num_leaves": 191, "learning_rate": 0.02, "min_child_samples": 80},
            "direct_params": {"num_leaves": 191, "learning_rate": 0.02, "min_child_samples": 80},
        },
        {
            "config_id": "lgb_v2_c5",
            "freq_params": {"num_leaves": 79, "learning_rate": 0.03, "subsample": 0.9},
            "sev_params": {"num_leaves": 127, "learning_rate": 0.03, "subsample": 0.9},
            "direct_params": {"num_leaves": 127, "learning_rate": 0.03, "subsample": 0.9},
        },
        {
            "config_id": "lgb_v2_c6",
            "freq_params": {"num_leaves": 55, "learning_rate": 0.05, "min_child_samples": 60},
            "sev_params": {"num_leaves": 111, "learning_rate": 0.05, "min_child_samples": 50},
            "direct_params": {"num_leaves": 111, "learning_rate": 0.05, "min_child_samples": 50},
        },
    ],
    "xgboost": [
        {
            "config_id": "xgb_v2_c1",
            "freq_params": {"max_depth": 6, "learning_rate": 0.03},
            "sev_params": {"max_depth": 8, "learning_rate": 0.03},
            "direct_params": {"max_depth": 8, "learning_rate": 0.03},
        },
        {
            "config_id": "xgb_v2_c2",
            "freq_params": {"max_depth": 5, "learning_rate": 0.04, "min_child_weight": 8},
            "sev_params": {"max_depth": 7, "learning_rate": 0.04, "min_child_weight": 8},
            "direct_params": {"max_depth": 7, "learning_rate": 0.04, "min_child_weight": 8},
        },
        {
            "config_id": "xgb_v2_c3",
            "freq_params": {"max_depth": 7, "learning_rate": 0.02, "min_child_weight": 4},
            "sev_params": {"max_depth": 9, "learning_rate": 0.02, "min_child_weight": 4},
            "direct_params": {"max_depth": 9, "learning_rate": 0.02, "min_child_weight": 4},
        },
        {
            "config_id": "xgb_v2_c4",
            "freq_params": {"max_depth": 6, "learning_rate": 0.02, "min_child_weight": 12},
            "sev_params": {"max_depth": 8, "learning_rate": 0.02, "min_child_weight": 10},
            "direct_params": {"max_depth": 8, "learning_rate": 0.02, "min_child_weight": 10},
        },
        {
            "config_id": "xgb_v2_c5",
            "freq_params": {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 6},
            "sev_params": {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 6},
            "direct_params": {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 6},
        },
        {
            "config_id": "xgb_v2_c6",
            "freq_params": {"max_depth": 8, "learning_rate": 0.02, "min_child_weight": 3},
            "sev_params": {"max_depth": 10, "learning_rate": 0.02, "min_child_weight": 3},
            "direct_params": {"max_depth": 10, "learning_rate": 0.02, "min_child_weight": 3},
        },
    ],
}


V2_SCREENING_FAMILIES: List[Dict[str, Any]] = [
    {"family": "two_part_classic", "severity_mode": "classic", "tweedie_power": 1.5},
    {"family": "two_part_classic", "severity_mode": "weighted_tail", "tweedie_power": 1.5},
    {"family": "two_part_classic", "severity_mode": "winsorized", "tweedie_power": 1.5},
    {"family": "two_part_tweedie", "severity_mode": "classic", "tweedie_power": 1.3},
    {"family": "two_part_tweedie", "severity_mode": "classic", "tweedie_power": 1.5},
    {"family": "two_part_tweedie", "severity_mode": "classic", "tweedie_power": 1.7},
    {"family": "direct_tweedie", "severity_mode": "classic", "tweedie_power": 1.5},
]
