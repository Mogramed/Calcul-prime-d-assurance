from __future__ import annotations

from functools import lru_cache
from typing import cast

from fastapi import Request

from insurance_pricing.api.audit import AuditStore
from insurance_pricing.api.service import PredictionService
from insurance_pricing.api.settings import AppSettings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def get_prediction_service(request: Request) -> PredictionService:
    return cast(PredictionService, request.app.state.prediction_service)


def get_audit_store(request: Request) -> AuditStore:
    return cast(AuditStore, request.app.state.audit_store)
