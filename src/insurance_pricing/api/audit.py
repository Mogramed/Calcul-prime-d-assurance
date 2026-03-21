from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class PredictionOutputRecord:
    record_position: int
    input_index: int | None
    frequency_prediction: float | None
    severity_prediction: float | None
    prime_prediction: float | None


@dataclass(frozen=True, slots=True)
class PredictionAuditRecord:
    request_id: str
    endpoint: str
    run_id: str
    record_count: int
    payload_hash: str
    status_code: int
    latency_ms: float
    outputs: Sequence[PredictionOutputRecord]


@dataclass(frozen=True, slots=True)
class ApiErrorAuditRecord:
    request_id: str
    endpoint: str
    run_id: str | None
    status_code: int
    exception_type: str
    message: str
    traceback_excerpt: str | None
    payload_hash: str | None


class AuditStoreUnavailableError(RuntimeError):
    """Raised when prediction persistence cannot complete."""


class AuditStore(Protocol):
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def check_ready(self) -> bool: ...

    async def persist_prediction(self, record: PredictionAuditRecord) -> None: ...

    async def persist_api_error(self, record: ApiErrorAuditRecord) -> None: ...


def hash_payload(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def hash_raw_payload(raw_body: bytes) -> str | None:
    if not raw_body:
        return None
    return sha256(raw_body).hexdigest()
