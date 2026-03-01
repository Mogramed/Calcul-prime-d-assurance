from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from src.insurance_pricing.inference.predict import (
    build_submission_from_run,
    predict_from_run as _predict_from_run,
    save_submission_from_run as _save_submission_from_run,
)


def predict_from_run(run_id: str, input_df: pd.DataFrame) -> pd.DataFrame:
    warnings.warn(
        "src.insurance_pricing.runtime.inference.predict_from_run is deprecated; "
        "use src.insurance_pricing.inference.predict.predict_from_run.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _predict_from_run(run_id, input_df)


def build_submission(run_id: str, test_df: pd.DataFrame) -> pd.DataFrame:
    warnings.warn(
        "src.insurance_pricing.runtime.inference.build_submission is deprecated; "
        "use src.insurance_pricing.inference.predict.build_submission_from_run.",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_submission_from_run(run_id, test_df)


def save_submission_from_run(run_id: str, test_df: pd.DataFrame, output_path: str | Path) -> Path:
    warnings.warn(
        "src.insurance_pricing.runtime.inference.save_submission_from_run is deprecated; "
        "use src.insurance_pricing.inference.predict.save_submission_from_run.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _save_submission_from_run(run_id, test_df, output_path)
