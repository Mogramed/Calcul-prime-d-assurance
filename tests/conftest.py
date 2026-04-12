from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from insurance_pricing.api import AppSettings
from insurance_pricing.api.account_emailing import AccountEmailDeliveryRecord, AccountEmailSender
from insurance_pricing.api.audit import (
    ApiErrorAuditRecord,
    AuditStore,
    AuditStoreUnavailableError,
    PredictionAuditRecord,
)
from insurance_pricing.api.auth_store import (
    AdminUserSummaryRecord,
    EmailVerificationCreateRecord,
    SessionCreateRecord,
    StoredAuthUserRecord,
    StoredUserRecord,
    UserAlreadyExistsError,
    UserStore,
)
from insurance_pricing.api.quote_emailing import QuoteEmailDeliveryRecord, QuoteEmailSender
from insurance_pricing.api.quote_store import (
    AdminQuoteSummaryRecord,
    QuoteCreateRecord,
    QuoteStore,
    QuoteStoreUnavailableError,
    QuoteSummaryRecord,
    QuoteUpdateRecord,
    StoredQuoteRecord,
)


def latest_run_id() -> str | None:
    registry_path = Path("artifacts/models/registry.csv")
    if not registry_path.exists():
        return None
    registry = pd.read_csv(registry_path)
    if registry.empty or "run_id" not in registry.columns:
        return None
    return str(registry.iloc[-1]["run_id"])


class InMemoryAuditStore(AuditStore, QuoteStore, UserStore):
    def __init__(
        self,
        *,
        ready: bool = True,
        fail_prediction_persistence: bool = False,
        fail_error_persistence: bool = False,
        fail_quote_persistence: bool = False,
    ) -> None:
        self.ready = ready
        self.fail_prediction_persistence = fail_prediction_persistence
        self.fail_error_persistence = fail_error_persistence
        self.fail_quote_persistence = fail_quote_persistence
        self.predictions: list[PredictionAuditRecord] = []
        self.errors: list[ApiErrorAuditRecord] = []
        self.quotes: list[StoredQuoteRecord] = []
        self.users: list[StoredAuthUserRecord] = []
        self.sessions: dict[str, tuple[str, datetime]] = {}
        self.email_verification_tokens: dict[str, tuple[str, datetime]] = {}
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

    async def create_user(self, record) -> StoredUserRecord:
        if any(user.email == record.email for user in self.users):
            raise UserAlreadyExistsError("User already exists.")
        stored_user = StoredAuthUserRecord(
            id=str(uuid4()),
            created_at_utc=datetime.now(UTC),
            email=record.email,
            password_hash=record.password_hash,
            role=record.role,
            is_active=True,
            email_verified_at_utc=record.email_verified_at_utc,
        )
        self.users.insert(0, stored_user)
        return _public_user(stored_user)

    async def get_user_auth_by_email(self, email: str) -> StoredAuthUserRecord | None:
        for user in self.users:
            if user.email == email:
                return user
        return None

    async def get_user_by_id(self, user_id: str) -> StoredUserRecord | None:
        for user in self.users:
            if user.id == user_id:
                return _public_user(user)
        return None

    async def get_user_by_session_token_hash(
        self, session_token_hash: str
    ) -> StoredUserRecord | None:
        session = self.sessions.get(session_token_hash)
        if session is None:
            return None
        user_id, expires_at_utc = session
        if expires_at_utc <= datetime.now(UTC):
            return None
        return await self.get_user_by_id(user_id)

    async def create_session(self, record: SessionCreateRecord) -> None:
        self.sessions[record.token_hash] = (record.user_id, record.expires_at_utc)

    async def delete_session(self, session_token_hash: str) -> None:
        self.sessions.pop(session_token_hash, None)

    async def create_email_verification(self, record: EmailVerificationCreateRecord) -> None:
        self.email_verification_tokens[record.token_hash] = (record.user_id, record.expires_at_utc)

    async def verify_email(self, token_hash: str) -> StoredUserRecord | None:
        verification = self.email_verification_tokens.pop(token_hash, None)
        if verification is None:
            return None
        user_id, expires_at_utc = verification
        if expires_at_utc <= datetime.now(UTC):
            return None
        for index, user in enumerate(self.users):
            if user.id == user_id:
                updated = StoredAuthUserRecord(
                    id=user.id,
                    created_at_utc=user.created_at_utc,
                    email=user.email,
                    password_hash=user.password_hash,
                    role=user.role,
                    is_active=user.is_active,
                    email_verified_at_utc=datetime.now(UTC),
                )
                self.users[index] = updated
                return _public_user(updated)
        return None

    async def list_admin_users(self) -> list[AdminUserSummaryRecord]:
        return [
            AdminUserSummaryRecord(
                id=user.id,
                created_at_utc=user.created_at_utc,
                email=user.email,
                role=user.role,
                is_active=user.is_active,
                email_verified_at_utc=user.email_verified_at_utc,
            )
            for user in sorted(self.users, key=lambda item: item.created_at_utc, reverse=True)
        ]

    async def deactivate_user(self, user_id: str) -> StoredUserRecord | None:
        for index, user in enumerate(self.users):
            if user.id == user_id:
                updated = StoredAuthUserRecord(
                    id=user.id,
                    created_at_utc=user.created_at_utc,
                    email=user.email,
                    password_hash=user.password_hash,
                    role=user.role,
                    is_active=False,
                    email_verified_at_utc=user.email_verified_at_utc,
                )
                self.users[index] = updated
                self.sessions = {
                    token_hash: value
                    for token_hash, value in self.sessions.items()
                    if value[0] != user_id
                }
                return _public_user(updated)
        return None

    async def attach_quotes_to_user(self, *, client_id_hash: str, user_id: str) -> int:
        attached = 0
        for index, quote in enumerate(self.quotes):
            if (
                quote.client_id_hash == client_id_hash
                and quote.user_id is None
                and quote.deleted_at_utc is None
            ):
                self.quotes[index] = StoredQuoteRecord(
                    id=quote.id,
                    created_at_utc=quote.created_at_utc,
                    client_id_hash=quote.client_id_hash,
                    user_id=user_id,
                    run_id=quote.run_id,
                    input_payload=dict(quote.input_payload),
                    frequency_prediction=quote.frequency_prediction,
                    severity_prediction=quote.severity_prediction,
                    prime_prediction=quote.prime_prediction,
                    deleted_at_utc=quote.deleted_at_utc,
                )
                attached += 1
        return attached

    async def create_quote(self, record: QuoteCreateRecord) -> StoredQuoteRecord:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        stored_quote = StoredQuoteRecord(
            id=str(uuid4()),
            created_at_utc=datetime.now(UTC),
            client_id_hash=record.client_id_hash,
            user_id=record.user_id,
            run_id=record.run_id,
            input_payload=dict(record.input_payload),
            frequency_prediction=record.frequency_prediction,
            severity_prediction=record.severity_prediction,
            prime_prediction=record.prime_prediction,
            deleted_at_utc=None,
        )
        self.quotes.insert(0, stored_quote)
        return stored_quote

    async def update_quote(self, record: QuoteUpdateRecord) -> StoredQuoteRecord | None:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        for index, quote in enumerate(self.quotes):
            if quote.id == record.quote_id and quote.deleted_at_utc is None:
                updated = StoredQuoteRecord(
                    id=quote.id,
                    created_at_utc=quote.created_at_utc,
                    client_id_hash=quote.client_id_hash,
                    user_id=record.user_id,
                    run_id=record.run_id,
                    input_payload=dict(record.input_payload),
                    frequency_prediction=record.frequency_prediction,
                    severity_prediction=record.severity_prediction,
                    prime_prediction=record.prime_prediction,
                    deleted_at_utc=quote.deleted_at_utc,
                )
                self.quotes[index] = updated
                return updated
        return None

    async def list_quotes(self, client_id_hash: str) -> list[QuoteSummaryRecord]:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        return [
            _quote_summary(quote)
            for quote in self.quotes
            if quote.client_id_hash == client_id_hash and quote.deleted_at_utc is None
        ]

    async def list_user_quotes(self, user_id: str) -> list[QuoteSummaryRecord]:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        return [
            _quote_summary(quote)
            for quote in self.quotes
            if quote.user_id == user_id and quote.deleted_at_utc is None
        ]

    async def get_quote(
        self,
        *,
        quote_id: str,
        client_id_hash: str,
    ) -> StoredQuoteRecord | None:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        for quote in self.quotes:
            if (
                quote.id == quote_id
                and quote.client_id_hash == client_id_hash
                and quote.deleted_at_utc is None
            ):
                return quote
        return None

    async def get_user_quote(self, *, quote_id: str, user_id: str) -> StoredQuoteRecord | None:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        for quote in self.quotes:
            if quote.id == quote_id and quote.user_id == user_id and quote.deleted_at_utc is None:
                return quote
        return None

    async def get_any_quote(self, quote_id: str) -> StoredQuoteRecord | None:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        for quote in self.quotes:
            if quote.id == quote_id:
                return quote
        return None

    async def list_admin_quotes(self, *, limit: int = 100) -> list[AdminQuoteSummaryRecord]:
        recent_quotes = self.quotes[:limit]
        return [
            AdminQuoteSummaryRecord(
                id=quote.id,
                created_at_utc=quote.created_at_utc,
                run_id=quote.run_id,
                type_contrat=str(quote.input_payload.get("type_contrat", "")),
                marque_vehicule=str(quote.input_payload.get("marque_vehicule", "")),
                modele_vehicule=str(quote.input_payload.get("modele_vehicule", "")),
                prime_prediction=quote.prime_prediction,
                user_id=quote.user_id,
                owner_email=_owner_email(self.users, quote.user_id),
                deleted_at_utc=quote.deleted_at_utc,
            )
            for quote in recent_quotes
        ]

    async def delete_quote(self, quote_id: str) -> StoredQuoteRecord | None:
        if self.fail_quote_persistence or not self.ready:
            raise QuoteStoreUnavailableError("Quote persistence is unavailable.")
        for index, quote in enumerate(self.quotes):
            if quote.id == quote_id:
                deleted = StoredQuoteRecord(
                    id=quote.id,
                    created_at_utc=quote.created_at_utc,
                    client_id_hash=quote.client_id_hash,
                    user_id=quote.user_id,
                    run_id=quote.run_id,
                    input_payload=dict(quote.input_payload),
                    frequency_prediction=quote.frequency_prediction,
                    severity_prediction=quote.severity_prediction,
                    prime_prediction=quote.prime_prediction,
                    deleted_at_utc=datetime.now(UTC),
                )
                self.quotes[index] = deleted
                return deleted
        return None


@dataclass(frozen=True, slots=True)
class SentQuoteEmail:
    quote_id: str
    recipient_email: str
    pdf_bytes: bytes


class InMemoryQuoteEmailSender(QuoteEmailSender):
    def __init__(self, *, fail_send: bool = False) -> None:
        self.fail_send = fail_send
        self.sent_emails: list[SentQuoteEmail] = []

    async def send_quote_email(
        self,
        *,
        quote: StoredQuoteRecord,
        recipient_email: str,
        pdf_bytes: bytes,
    ) -> QuoteEmailDeliveryRecord:
        if self.fail_send:
            raise RuntimeError("Quote email delivery failed.")
        self.sent_emails.append(
            SentQuoteEmail(
                quote_id=quote.id,
                recipient_email=recipient_email,
                pdf_bytes=pdf_bytes,
            )
        )
        return QuoteEmailDeliveryRecord(
            status="sent",
            recipient_email=recipient_email,
            detail="Email accepted by the in-memory sender.",
            provider_status_code=202,
        )


@dataclass(frozen=True, slots=True)
class SentVerificationEmail:
    recipient_email: str
    verification_token: str


class InMemoryAccountEmailSender(AccountEmailSender):
    def __init__(self, *, fail_send: bool = False) -> None:
        self.fail_send = fail_send
        self.sent_emails: list[SentVerificationEmail] = []

    async def send_verification_email(
        self,
        *,
        recipient_email: str,
        verification_token: str,
    ) -> AccountEmailDeliveryRecord:
        if self.fail_send:
            raise RuntimeError("Account email delivery failed.")
        self.sent_emails.append(
            SentVerificationEmail(
                recipient_email=recipient_email,
                verification_token=verification_token,
            )
        )
        return AccountEmailDeliveryRecord(
            status="sent",
            recipient_email=recipient_email,
            detail="Verification email accepted by the in-memory sender.",
            provider_status_code=202,
        )


def _public_user(user: StoredAuthUserRecord) -> StoredUserRecord:
    return StoredUserRecord(
        id=user.id,
        created_at_utc=user.created_at_utc,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        email_verified_at_utc=user.email_verified_at_utc,
    )


def _quote_summary(quote: StoredQuoteRecord) -> QuoteSummaryRecord:
    return QuoteSummaryRecord(
        id=quote.id,
        created_at_utc=quote.created_at_utc,
        run_id=quote.run_id,
        type_contrat=str(quote.input_payload["type_contrat"]),
        marque_vehicule=str(quote.input_payload["marque_vehicule"]),
        modele_vehicule=str(quote.input_payload["modele_vehicule"]),
        prime_prediction=quote.prime_prediction,
    )


def _owner_email(users: list[StoredAuthUserRecord], user_id: str | None) -> str | None:
    if user_id is None:
        return None
    for user in users:
        if user.id == user_id:
            return user.email
    return None


@pytest.fixture(scope="session")
def existing_run_id() -> str:
    run_id = latest_run_id()
    if run_id is None:
        pytest.skip("No trained model registry found.")
    return run_id


@pytest.fixture(scope="session")
def sample_prediction_records() -> list[dict]:
    sample = (
        pd.read_csv("data/test.csv").head(2).drop(columns=["index"], errors="ignore").fillna("")
    )
    return json.loads(sample.to_json(orient="records"))


@pytest.fixture
def api_settings(existing_run_id: str) -> AppSettings:
    return AppSettings(
        run_id=existing_run_id,
        database_url="postgresql+psycopg://unit-test:unit-test@localhost/unit-test",
        log_level="INFO",
        log_json=True,
        admin_emails=["admin@nova-assurances.fr"],
    )


@pytest.fixture
def in_memory_audit_store() -> InMemoryAuditStore:
    return InMemoryAuditStore()


@pytest.fixture
def audit_store_factory():
    return InMemoryAuditStore


@pytest.fixture
def in_memory_quote_email_sender() -> InMemoryQuoteEmailSender:
    return InMemoryQuoteEmailSender()


@pytest.fixture
def in_memory_account_email_sender() -> InMemoryAccountEmailSender:
    return InMemoryAccountEmailSender()
