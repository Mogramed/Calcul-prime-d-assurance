from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from insurance_pricing._typing import FloatArray, as_float_array
from insurance_pricing.models.calibration import apply_calibrator
from insurance_pricing.models.frequency import FrequencyModel
from insurance_pricing.models.severity import SeverityModel
from insurance_pricing.models.tail import apply_tail_mapper


@dataclass
class PrimeModel:
    freq_model: FrequencyModel
    sev_model: SeverityModel
    calibration_method: str = "none"
    calibrator: Any | None = None
    tail_mapper: dict[str, Any] | None = None
    non_negative: bool = True

    def predict_components(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        pred_freq_raw = self.freq_model.predict_proba(raw_df)
        pred_freq = apply_calibrator(self.calibrator, pred_freq_raw, self.calibration_method)
        pred_sev_raw = self.sev_model.predict(raw_df)
        pred_sev = apply_tail_mapper(self.tail_mapper, pred_sev_raw)
        pred_prime = pred_freq * pred_sev
        if self.non_negative:
            pred_prime = np.maximum(pred_prime, 0.0)
            pred_sev = np.maximum(pred_sev, 0.0)
            pred_freq = np.clip(pred_freq, 0.0, 1.0)
        return pd.DataFrame(
            {
                "pred_freq": pred_freq,
                "pred_sev": pred_sev,
                "pred_prime": pred_prime,
            }
        )

    def predict_prime(self, raw_df: pd.DataFrame) -> FloatArray:
        return as_float_array(self.predict_components(raw_df)["pred_prime"].to_numpy(dtype=float))
