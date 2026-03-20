from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from insurance_pricing.features.schema import build_feature_frame_for_inference


@dataclass
class FrequencyModel:
    model: Any
    feature_cols: list[str]
    cat_cols: list[str]
    engine: str = "catboost"

    def predict_proba(self, raw_df: pd.DataFrame) -> np.ndarray:
        X = build_feature_frame_for_inference(
            raw_df,
            feature_cols=self.feature_cols,
            cat_cols=self.cat_cols,
        )
        p = self.model.predict_proba(X)[:, 1]
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


def fit_frequency_model(
    X_train: pd.DataFrame,
    y_freq: np.ndarray,
    *,
    cat_cols: Sequence[str],
    engine: str = "catboost",
    seed: int = 42,
    params: Mapping[str, Any] | None = None,
) -> FrequencyModel:
    params = dict(params or {})
    e = str(engine).lower()
    if e != "catboost":
        raise ValueError("Only catboost engine is supported in the canonical train_run.")

    from catboost import CatBoostClassifier

    base = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_seed": int(seed),
        "verbose": False,
        "allow_writing_files": False,
    }
    base.update(params)
    model = CatBoostClassifier(**base)
    model.fit(X_train, y_freq, cat_features=[X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns])
    return FrequencyModel(
        model=model,
        feature_cols=list(X_train.columns),
        cat_cols=[c for c in cat_cols if c in X_train.columns],
        engine="catboost",
    )

