from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import pandas as pd

from insurance_pricing.data.schema import INDEX_COL
from insurance_pricing.runtime.persistence import load_model_bundle


class PredictionService:
    def __init__(
        self,
        *,
        run_id: str,
        freq_model: Any,
        sev_model: Any,
        prime_model: Any,
        manifest: Mapping[str, Any],
        feature_schema: Mapping[str, Any],
    ) -> None:
        self.run_id = run_id
        self.freq_model = freq_model
        self.sev_model = sev_model
        self.prime_model = prime_model
        self.manifest = dict(manifest)
        self.feature_schema = dict(feature_schema)
        self.metrics = dict(manifest.get("metrics", {}))

    @classmethod
    def load(cls, run_id: str) -> PredictionService:
        bundle = load_model_bundle(run_id)
        return cls(
            run_id=run_id,
            freq_model=bundle["freq_model"],
            sev_model=bundle["sev_model"],
            prime_model=bundle["prime_model"],
            manifest=bundle["manifest"],
            feature_schema=bundle["feature_schema"],
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "run_id": self.run_id,
            "model_loaded": True,
        }

    def predict_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict_records([record])[0]

    def predict_records(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        return self._predict_components(records)

    def predict_frequency_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict_frequency_records([record])[0]

    def predict_frequency_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            self._select_fields(prediction, {"index", "frequency_prediction"})
            for prediction in self._predict_components(records)
        ]

    def predict_severity_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return self.predict_severity_records([record])[0]

    def predict_severity_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            self._select_fields(prediction, {"index", "severity_prediction"})
            for prediction in self._predict_components(records)
        ]

    def current_model_metadata(self) -> dict[str, Any]:
        config = self.manifest.get("config", {})
        return {
            "run_id": self.run_id,
            "created_at_utc": self.manifest.get("created_at_utc"),
            "notes": self.manifest.get("notes"),
            "model_files": dict(self.manifest.get("model_files", {})),
            "metrics": dict(self.manifest.get("metrics", {})),
            "feature_schema": {
                "feature_count": len(self.feature_schema.get("feature_cols", [])),
                "categorical_feature_count": len(self.feature_schema.get("cat_cols", [])),
                "numerical_feature_count": len(self.feature_schema.get("num_cols", [])),
                "feature_columns": list(self.feature_schema.get("feature_cols", [])),
                "categorical_columns": list(self.feature_schema.get("cat_cols", [])),
                "numerical_columns": list(self.feature_schema.get("num_cols", [])),
            },
            "config": {
                "feature_set": config.get("feature_set"),
                "drop_identifiers": config.get("drop_identifiers"),
                "frequency_engine": config.get("freq", {}).get("engine"),
                "frequency_calibration": config.get("freq", {}).get("calibration"),
                "severity_engine": config.get("sev", {}).get("engine"),
                "severity_family": config.get("sev", {}).get("family"),
                "severity_mode": config.get("sev", {}).get("severity_mode"),
                "tweedie_power": config.get("sev", {}).get("tweedie_power"),
                "tail_mapper_enabled": config.get("sev", {}).get("use_tail_mapper"),
                "non_negative": config.get("prime", {}).get("non_negative"),
            },
        }

    def _predict_components(self, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        input_df = pd.DataFrame.from_records(records)
        prediction_df = self.prime_model.predict_components(input_df)
        if INDEX_COL in input_df.columns:
            prediction_df.insert(0, INDEX_COL, input_df[INDEX_COL].to_numpy())
        return [self._map_prediction(row) for row in prediction_df.to_dict(orient="records")]

    def _map_prediction(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "frequency_prediction": float(row["pred_freq"]),
            "severity_prediction": float(row["pred_sev"]),
            "prime_prediction": float(row["pred_prime"]),
        }
        if INDEX_COL in row and row[INDEX_COL] is not None and not pd.isna(row[INDEX_COL]):
            payload[INDEX_COL] = self._normalize_index(row[INDEX_COL])
        return payload

    @staticmethod
    def _select_fields(prediction: Mapping[str, Any], allowed_fields: set[str]) -> dict[str, Any]:
        return {key: value for key, value in prediction.items() if key in allowed_fields}

    @staticmethod
    def _normalize_index(value: Any) -> int:
        if isinstance(value, bool):
            raise TypeError("Boolean values are not valid indexes.")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and isfinite(value) and value.is_integer():
            return int(value)
        return int(value)
