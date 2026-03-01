from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.insurance_pricing.evaluation.metrics import make_tail_weights
from src.insurance_pricing.features.schema import build_feature_frame_for_inference
from src.insurance_pricing.features.target_encoding import _apply_winsor


@dataclass
class SeverityModel:
    model: Any
    feature_cols: list[str]
    cat_cols: list[str]
    family: str
    severity_mode: str
    tweedie_power: float = 1.3
    smear_factor: float = 1.0

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        X = build_feature_frame_for_inference(
            raw_df,
            feature_cols=self.feature_cols,
            cat_cols=self.cat_cols,
        )
        fam = self.family.lower()
        if fam == "two_part_tweedie":
            pred = self.model.predict(X)
            return np.maximum(np.asarray(pred, dtype=float), 0.0)
        z = self.model.predict(X)
        pred = np.maximum(self.smear_factor * np.exp(np.asarray(z, dtype=float)) - 1.0, 0.0)
        return np.maximum(np.asarray(pred, dtype=float), 0.0)


def _fit_catboost_severity(
    X_train: pd.DataFrame,
    y_sev: np.ndarray,
    y_freq: np.ndarray,
    *,
    cat_cols: Sequence[str],
    family: str,
    severity_mode: str,
    tweedie_power: float,
    seed: int,
    params: Mapping[str, Any],
) -> SeverityModel:
    from catboost import CatBoostRegressor

    pos = np.asarray(y_freq, dtype=int) == 1
    X_pos = X_train.loc[pos].copy()
    y_pos = np.clip(np.asarray(y_sev, dtype=float)[pos], 0.0, None)
    if int(len(y_pos)) < 20:
        raise ValueError("Not enough positive severity samples.")

    sev_mode = str(severity_mode).lower()
    if sev_mode == "winsorized":
        y_pos_fit = _apply_winsor(y_pos, quantile=0.995)
    else:
        y_pos_fit = y_pos
    sample_weight = make_tail_weights(y_pos_fit) if sev_mode == "weighted_tail" else None

    p = {
        "iterations": 3500,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_seed": int(seed),
        "verbose": False,
        "allow_writing_files": False,
    }
    p.update(dict(params))

    fam = str(family).lower()
    if fam == "two_part_tweedie":
        p["loss_function"] = f"Tweedie:variance_power={float(tweedie_power):.4f}"
        p.setdefault("eval_metric", "RMSE")
        reg = CatBoostRegressor(**p)
        reg.fit(
            X_pos,
            y_pos_fit,
            cat_features=[X_pos.columns.get_loc(c) for c in cat_cols if c in X_pos.columns],
            sample_weight=sample_weight,
        )
        return SeverityModel(
            model=reg,
            feature_cols=list(X_train.columns),
            cat_cols=[c for c in cat_cols if c in X_train.columns],
            family=fam,
            severity_mode=sev_mode,
            tweedie_power=float(tweedie_power),
            smear_factor=1.0,
        )

    p["loss_function"] = "RMSE"
    p.setdefault("eval_metric", "RMSE")
    reg = CatBoostRegressor(**p)
    y_log = np.log1p(y_pos_fit)
    reg.fit(
        X_pos,
        y_log,
        cat_features=[X_pos.columns.get_loc(c) for c in cat_cols if c in X_pos.columns],
        sample_weight=sample_weight,
    )
    z_tr = reg.predict(X_pos)
    resid = y_log - np.asarray(z_tr, dtype=float)
    if sample_weight is not None:
        smear = float(np.average(np.exp(resid), weights=np.asarray(sample_weight, dtype=float)))
    else:
        smear = float(np.mean(np.exp(resid)))
    if not np.isfinite(smear) or smear <= 0:
        smear = 1.0
    return SeverityModel(
        model=reg,
        feature_cols=list(X_train.columns),
        cat_cols=[c for c in cat_cols if c in X_train.columns],
        family="two_part_classic",
        severity_mode=sev_mode,
        tweedie_power=float(tweedie_power),
        smear_factor=float(smear),
    )


def fit_severity_model(
    X_train: pd.DataFrame,
    y_sev: np.ndarray,
    y_freq: np.ndarray,
    *,
    cat_cols: Sequence[str],
    engine: str = "catboost",
    family: str = "two_part_tweedie",
    severity_mode: str = "weighted_tail",
    tweedie_power: float = 1.3,
    seed: int = 42,
    params: Mapping[str, Any] | None = None,
) -> SeverityModel:
    e = str(engine).lower()
    if e != "catboost":
        raise ValueError("Only catboost engine is supported in the canonical train_run.")
    return _fit_catboost_severity(
        X_train=X_train,
        y_sev=y_sev,
        y_freq=y_freq,
        cat_cols=cat_cols,
        family=family,
        severity_mode=severity_mode,
        tweedie_power=tweedie_power,
        seed=seed,
        params=dict(params or {}),
    )

