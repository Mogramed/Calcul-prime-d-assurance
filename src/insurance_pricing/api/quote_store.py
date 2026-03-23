from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Protocol

CLIENT_ID_HEADER = "X-Client-ID"


@dataclass(frozen=True, slots=True)
class QuoteCreateRecord:
    client_id_hash: str
    user_id: str | None
    run_id: str
    input_payload: Mapping[str, Any]
    frequency_prediction: float
    severity_prediction: float
    prime_prediction: float


@dataclass(frozen=True, slots=True)
class StoredQuoteRecord:
    id: str
    created_at_utc: datetime
    client_id_hash: str
    user_id: str | None
    run_id: str
    input_payload: dict[str, Any]
    frequency_prediction: float
    severity_prediction: float
    prime_prediction: float
    deleted_at_utc: datetime | None


@dataclass(frozen=True, slots=True)
class QuoteSummaryRecord:
    id: str
    created_at_utc: datetime
    run_id: str
    input_index: int | None
    type_contrat: str
    marque_vehicule: str
    modele_vehicule: str
    prime_prediction: float


@dataclass(frozen=True, slots=True)
class AdminQuoteSummaryRecord:
    id: str
    created_at_utc: datetime
    run_id: str
    type_contrat: str
    marque_vehicule: str
    modele_vehicule: str
    prime_prediction: float
    user_id: str | None
    owner_email: str | None
    deleted_at_utc: datetime | None


class QuoteStoreUnavailableError(RuntimeError):
    """Raised when quote persistence cannot complete."""


class QuoteStore(Protocol):
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def check_ready(self) -> bool: ...

    async def create_quote(self, record: QuoteCreateRecord) -> StoredQuoteRecord: ...

    async def list_quotes(self, client_id_hash: str) -> Sequence[QuoteSummaryRecord]: ...

    async def list_user_quotes(self, user_id: str) -> Sequence[QuoteSummaryRecord]: ...

    async def get_quote(
        self,
        *,
        quote_id: str,
        client_id_hash: str,
    ) -> StoredQuoteRecord | None: ...

    async def get_user_quote(
        self,
        *,
        quote_id: str,
        user_id: str,
    ) -> StoredQuoteRecord | None: ...

    async def get_any_quote(self, quote_id: str) -> StoredQuoteRecord | None: ...

    async def list_admin_quotes(self, *, limit: int = 100) -> Sequence[AdminQuoteSummaryRecord]: ...

    async def delete_quote(self, quote_id: str) -> StoredQuoteRecord | None: ...


def hash_client_id(client_id: str) -> str:
    return sha256(client_id.strip().lower().encode("utf-8")).hexdigest()
