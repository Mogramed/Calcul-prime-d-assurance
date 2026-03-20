from __future__ import annotations

from pathlib import Path

import pandas as pd

from insurance_pricing.data.io import load_train_test

DEFAULT_DS_DIR = Path("artifacts") / "ds"


def load_project_datasets(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_train_test(data_dir)

