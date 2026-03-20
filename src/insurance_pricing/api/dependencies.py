from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from insurance_pricing.api.service import PredictionService
from insurance_pricing.api.settings import AppSettings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service
