from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from insurance_pricing.api import AppSettings
from insurance_pricing.api.audit import (
    ApiErrorAuditRecord,
    AuditStore,
    AuditStoreUnavailableError,
    PredictionAuditRecord,
)


def latest_run_id() -> str | None:
    registry_path = Path("artifacts/models/registry.csv")
    if not registry_path.exists():
        return None
    registry = pd.read_csv(registry_path)
    if registry.empty or "run_id" not in registry.columns:
        return None
    return str(registry.iloc[-1]["run_id"])


class InMemoryAuditStore(AuditStore):
    def __init__(
        self,
        *,
        ready: bool = True,
        fail_prediction_persistence: bool = False,
        fail_error_persistence: bool = False,
    ) -> None:
        self.ready = ready
        self.fail_prediction_persistence = fail_prediction_persistence
        self.fail_error_persistence = fail_error_persistence
        self.predictions: list[PredictionAuditRecord] = []
        self.errors: list[ApiErrorAuditRecord] = []
        self.started = False

    async def startup(self) -> None:
        self.started = True
        if not self.ready:
            raise AuditStoreUnavailableError("Audit store is not ready.")

    async def shutdown(self) -> None:
        self.started = False

    async def check_ready(self) -> bool:
        return self.ready

    async def persist_prediction(self, record: PredictionAuditRecord) -> None:
        if self.fail_prediction_persistence or not self.ready:
            raise AuditStoreUnavailableError("Prediction persistence is unavailable.")
        self.predictions.append(record)

    async def persist_api_error(self, record: ApiErrorAuditRecord) -> None:
        if self.fail_error_persistence:
            raise AuditStoreUnavailableError("Error persistence is unavailable.")
        self.errors.append(record)


@pytest.fixture(scope="session")
def existing_run_id() -> str:
    run_id = latest_run_id()
    if run_id is None:
        pytest.skip("No trained model registry found.")
    return run_id


@pytest.fixture(scope="session")
def sample_prediction_records() -> list[dict]:
    sample = pd.read_csv("data/test.csv").head(2).fillna("")
    return json.loads(sample.to_json(orient="records"))


@pytest.fixture
def api_settings(existing_run_id: str) -> AppSettings:
    return AppSettings(
        run_id=existing_run_id,
        database_url="postgresql+psycopg://unit-test:unit-test@localhost/unit-test",
        log_level="INFO",
        log_json=True,
    )


@pytest.fixture
def in_memory_audit_store() -> InMemoryAuditStore:
    return InMemoryAuditStore()


@pytest.fixture
def audit_store_factory():
    return InMemoryAuditStore
