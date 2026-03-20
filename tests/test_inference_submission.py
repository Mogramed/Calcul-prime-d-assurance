from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from insurance_pricing.inference.predict import build_submission_from_run


def _latest_run_id_from_registry() -> str | None:
    reg = Path("artifacts/models/registry.csv")
    if not reg.exists():
        return None
    df = pd.read_csv(reg)
    if df.empty or "run_id" not in df.columns:
        return None
    return str(df.iloc[-1]["run_id"])


def test_submission_shape_and_columns_if_model_exists():
    run_id = _latest_run_id_from_registry()
    if run_id is None:
        pytest.skip("No trained model registry found.")
    test_df = pd.read_csv("data/test.csv")
    sub = build_submission_from_run(run_id, test_df)
    assert list(sub.columns) == ["index", "pred"]
    assert len(sub) == len(test_df)
    assert (sub["pred"] >= 0).all()
