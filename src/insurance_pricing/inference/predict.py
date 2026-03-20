from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from insurance_pricing.data.schema import INDEX_COL
from insurance_pricing.inference.submission import build_submission as build_submission_df
from insurance_pricing.runtime.persistence import load_model_bundle


def predict_from_run(run_id: str, input_df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle(run_id)
    prime_model = bundle["prime_model"]
    pred_df = prime_model.predict_components(input_df)
    out = pred_df.copy()
    if INDEX_COL in input_df.columns:
        out.insert(0, INDEX_COL, input_df[INDEX_COL].to_numpy())
    return out


def build_submission_from_run(run_id: str, test_df: pd.DataFrame) -> pd.DataFrame:
    pred = predict_from_run(run_id, test_df)
    if INDEX_COL in pred.columns:
        idx = pred[INDEX_COL]
    elif INDEX_COL in test_df.columns:
        idx = test_df[INDEX_COL]
    else:
        idx = pd.Series(np.arange(len(test_df), dtype=int), name=INDEX_COL)
    return build_submission_df(pd.Series(idx), pred["pred_prime"].to_numpy(dtype=float))


def save_submission_from_run(run_id: str, test_df: pd.DataFrame, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = build_submission_from_run(run_id, test_df)
    sub.to_csv(out, index=False)
    return out

