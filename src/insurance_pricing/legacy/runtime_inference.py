from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from insurance_pricing.inference.predict import (
    build_submission_from_run,
)
from insurance_pricing.inference.predict import (
    predict_from_run as _predict_from_run,
)
from insurance_pricing.inference.predict import (
    save_submission_from_run as _save_submission_from_run,
)


def predict_from_run(run_id: str, input_df: pd.DataFrame) -> pd.DataFrame:
    warnings.warn(
        "insurance_pricing.runtime.inference.predict_from_run is deprecated; "
        "use insurance_pricing.inference.predict.predict_from_run.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _predict_from_run(run_id, input_df)


def build_submission(run_id: str, test_df: pd.DataFrame) -> pd.DataFrame:
    warnings.warn(
        "insurance_pricing.runtime.inference.build_submission is deprecated; "
        "use insurance_pricing.inference.predict.build_submission_from_run.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_submission_from_run(run_id, test_df)


def save_submission_from_run(run_id: str, test_df: pd.DataFrame, output_path: str | Path) -> Path:
    warnings.warn(
        "insurance_pricing.runtime.inference.save_submission_from_run is deprecated; "
        "use insurance_pricing.inference.predict.save_submission_from_run.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _save_submission_from_run(run_id, test_df, output_path)
