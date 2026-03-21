from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

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
        self.encoder: Optional[Any] = None

    def fit(self, X: pd.DataFrame) -> "OrdinalFrameEncoder":
        if not self.cat_cols:
            return self
        from sklearn.preprocessing import OrdinalEncoder

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

