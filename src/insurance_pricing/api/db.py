from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    delete,
    func,
    outerjoin,
    select,
    text,
    update,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

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
    UserCreateRecord,
    UserStore,
    UserStoreUnavailableError,
)
from insurance_pricing.api.quote_store import (
    AdminQuoteSummaryRecord,
    QuoteCreateRecord,
    QuoteStore,
    QuoteStoreUnavailableError,
    QuoteSummaryRecord,
    QuoteUpdateRecord,
    StoredQuoteRecord,
)


class Base(DeclarativeBase):
    pass


class PredictionRequestRow(Base):
    __tablename__ = "prediction_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    request_id: Mapped[str] = mapped_column(String(255), index=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    endpoint: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    outputs: Mapped[list[PredictionOutputRow]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",
    )


class PredictionOutputRow(Base):
    __tablename__ = "prediction_outputs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    prediction_request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("prediction_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    record_position: Mapped[int] = mapped_column(Integer, nullable=False)
    frequency_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    prime_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)

    request: Mapped[PredictionRequestRow] = relationship(back_populates="outputs")


class ApiErrorRow(Base):
    __tablename__ = "api_errors"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    request_id: Mapped[str] = mapped_column(String(255), index=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    endpoint: Mapped[str] = mapped_column(String(128), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    exception_type: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    traceback_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    email_verified_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    quotes: Mapped[list[QuoteRow]] = relationship(back_populates="user")
    sessions: Mapped[list[AuthSessionRow]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    email_verification_tokens: Mapped[list[EmailVerificationTokenRow]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class AuthSessionRow(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped[UserRow] = relationship(back_populates="sessions")


class EmailVerificationTokenRow(Base):
    __tablename__ = "email_verification_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[UserRow] = relationship(back_populates="email_verification_tokens")


class QuoteRow(Base):
    __tablename__ = "quotes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    client_id_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    run_id: Mapped[str] = mapped_column(String(255), nullable=False)
    input_payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    frequency_prediction: Mapped[float] = mapped_column(Float, nullable=False)
    severity_prediction: Mapped[float] = mapped_column(Float, nullable=False)
    prime_prediction: Mapped[float] = mapped_column(Float, nullable=False)
    deleted_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[UserRow | None] = relationship(back_populates="quotes")


class PostgresAuditStore(AuditStore, QuoteStore, UserStore):
    def __init__(self, database_url: str) -> None:
        _ensure_windows_selector_event_loop_policy()
        self.database_url = database_url
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    async def startup(self) -> None:
        self._engine = create_async_engine(self.database_url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        await self._ensure_database_connection()
        await self._ensure_migrations_applied()

    async def shutdown(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._sessionmaker = None

    async def check_ready(self) -> bool:
        if self._engine is None:
            return False
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise AuditStoreUnavailableError("PostgreSQL audit store is not initialized.")
        session = self._sessionmaker()
        try:
            yield session
        finally:
            await session.close()

    async def persist_prediction(self, record: PredictionAuditRecord) -> None:
        request_row = PredictionRequestRow(
            request_id=record.request_id,
            endpoint=record.endpoint,
            run_id=record.run_id,
            record_count=record.record_count,
            payload_hash=record.payload_hash,
            status_code=record.status_code,
            latency_ms=record.latency_ms,
        )
        output_rows = [
            PredictionOutputRow(
                record_position=output.record_position,
                frequency_prediction=output.frequency_prediction,
                severity_prediction=output.severity_prediction,
                prime_prediction=output.prime_prediction,
            )
            for output in record.outputs
        ]

        try:
            async with self.session() as session, session.begin():
                session.add(request_row)
                await session.flush()
                for output_row in output_rows:
                    output_row.prediction_request_id = request_row.id
                    session.add(output_row)
        except SQLAlchemyError as exc:
            raise AuditStoreUnavailableError("Prediction persistence failed.") from exc

    async def persist_api_error(self, record: ApiErrorAuditRecord) -> None:
        try:
            async with self.session() as session, session.begin():
                session.add(
                    ApiErrorRow(
                        request_id=record.request_id,
                        endpoint=record.endpoint,
                        run_id=record.run_id,
                        status_code=record.status_code,
                        exception_type=record.exception_type,
                        message=record.message,
                        traceback_excerpt=record.traceback_excerpt,
                        payload_hash=record.payload_hash,
                    )
                )
        except SQLAlchemyError as exc:
            raise AuditStoreUnavailableError("API error persistence failed.") from exc

    async def create_user(self, record: UserCreateRecord) -> StoredUserRecord:
        user_row = UserRow(
            email=record.email,
            password_hash=record.password_hash,
            role=record.role,
            is_active=True,
            email_verified_at_utc=record.email_verified_at_utc,
        )

        try:
            async with self.session() as session, session.begin():
                session.add(user_row)
                await session.flush()
                await session.refresh(user_row)
        except IntegrityError as exc:
            raise UserAlreadyExistsError("A user with this email already exists.") from exc
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("User persistence failed.") from exc
        return _stored_user_from_row(user_row)

    async def get_user_auth_by_email(self, email: str) -> StoredAuthUserRecord | None:
        try:
            async with self.session() as session:
                result = await session.execute(select(UserRow).where(UserRow.email == email))
                row = result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("User lookup failed.") from exc
        if row is None:
            return None
        return _stored_auth_user_from_row(row)

    async def get_user_by_id(self, user_id: str) -> StoredUserRecord | None:
        try:
            async with self.session() as session:
                row = await session.get(UserRow, user_id)
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("User lookup failed.") from exc
        if row is None:
            return None
        return _stored_user_from_row(row)

    async def get_user_by_session_token_hash(
        self, session_token_hash: str
    ) -> StoredUserRecord | None:
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(UserRow)
                    .join(AuthSessionRow, AuthSessionRow.user_id == UserRow.id)
                    .where(
                        AuthSessionRow.token_hash == session_token_hash,
                        AuthSessionRow.expires_at_utc > datetime.now(UTC),
                        UserRow.is_active.is_(True),
                    )
                )
                row = result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("Session lookup failed.") from exc
        if row is None:
            return None
        return _stored_user_from_row(row)

    async def create_session(self, record: SessionCreateRecord) -> None:
        try:
            async with self.session() as session, session.begin():
                session.add(
                    AuthSessionRow(
                        user_id=record.user_id,
                        token_hash=record.token_hash,
                        expires_at_utc=record.expires_at_utc,
                    )
                )
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("Session creation failed.") from exc

    async def create_email_verification(self, record: EmailVerificationCreateRecord) -> None:
        try:
            async with self.session() as session, session.begin():
                session.add(
                    EmailVerificationTokenRow(
                        user_id=record.user_id,
                        token_hash=record.token_hash,
                        expires_at_utc=record.expires_at_utc,
                    )
                )
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("Email verification persistence failed.") from exc

    async def verify_email(self, token_hash: str) -> StoredUserRecord | None:
        try:
            async with self.session() as session, session.begin():
                result = await session.execute(
                    select(EmailVerificationTokenRow, UserRow)
                    .join(UserRow, UserRow.id == EmailVerificationTokenRow.user_id)
                    .where(
                        EmailVerificationTokenRow.token_hash == token_hash,
                        EmailVerificationTokenRow.used_at_utc.is_(None),
                        EmailVerificationTokenRow.expires_at_utc > datetime.now(UTC),
                        UserRow.is_active.is_(True),
                    )
                )
                row = result.one_or_none()
                if row is None:
                    return None
                token_row, user_row = row
                if user_row.email_verified_at_utc is None:
                    user_row.email_verified_at_utc = datetime.now(UTC)
                token_row.used_at_utc = datetime.now(UTC)
                await session.flush()
                await session.refresh(user_row)
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("Email verification failed.") from exc
        return _stored_user_from_row(user_row)

    async def delete_session(self, session_token_hash: str) -> None:
        try:
            async with self.session() as session, session.begin():
                await session.execute(
                    delete(AuthSessionRow).where(AuthSessionRow.token_hash == session_token_hash)
                )
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("Session deletion failed.") from exc

    async def list_admin_users(self) -> list[AdminUserSummaryRecord]:
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(UserRow).order_by(UserRow.created_at_utc.desc(), UserRow.email.asc())
                )
                rows = result.scalars().all()
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("User listing failed.") from exc
        return [_admin_user_summary_from_row(row) for row in rows]

    async def deactivate_user(self, user_id: str) -> StoredUserRecord | None:
        try:
            async with self.session() as session, session.begin():
                row = await session.get(UserRow, user_id)
                if row is None:
                    return None
                row.is_active = False
                await session.execute(
                    delete(AuthSessionRow).where(AuthSessionRow.user_id == user_id)
                )
                await session.flush()
                await session.refresh(row)
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("User deactivation failed.") from exc
        return _stored_user_from_row(row)

    async def attach_quotes_to_user(self, *, client_id_hash: str, user_id: str) -> int:
        try:
            async with self.session() as session, session.begin():
                result = await session.execute(
                    update(QuoteRow)
                    .where(
                        QuoteRow.client_id_hash == client_id_hash,
                        QuoteRow.user_id.is_(None),
                        QuoteRow.deleted_at_utc.is_(None),
                    )
                    .values(user_id=user_id)
                )
                rowcount = getattr(result, "rowcount", None)
                return int(rowcount or 0)
        except SQLAlchemyError as exc:
            raise UserStoreUnavailableError("Quote attachment failed.") from exc

    async def create_quote(self, record: QuoteCreateRecord) -> StoredQuoteRecord:
        quote_row = QuoteRow(
            client_id_hash=record.client_id_hash,
            user_id=record.user_id,
            run_id=record.run_id,
            input_payload_json=dict(record.input_payload),
            frequency_prediction=record.frequency_prediction,
            severity_prediction=record.severity_prediction,
            prime_prediction=record.prime_prediction,
        )

        try:
            async with self.session() as session, session.begin():
                session.add(quote_row)
                await session.flush()
                await session.refresh(quote_row)
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote persistence failed.") from exc
        return _stored_quote_from_row(quote_row)

    async def update_quote(self, record: QuoteUpdateRecord) -> StoredQuoteRecord | None:
        try:
            async with self.session() as session, session.begin():
                row = await session.get(QuoteRow, record.quote_id)
                if row is None or row.deleted_at_utc is not None:
                    return None
                row.user_id = record.user_id
                row.run_id = record.run_id
                row.input_payload_json = dict(record.input_payload)
                row.frequency_prediction = record.frequency_prediction
                row.severity_prediction = record.severity_prediction
                row.prime_prediction = record.prime_prediction
                await session.flush()
                await session.refresh(row)
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote update failed.") from exc
        return _stored_quote_from_row(row)

    async def list_quotes(self, client_id_hash: str) -> list[QuoteSummaryRecord]:
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(QuoteRow)
                    .where(
                        QuoteRow.client_id_hash == client_id_hash,
                        QuoteRow.deleted_at_utc.is_(None),
                    )
                    .order_by(QuoteRow.created_at_utc.desc())
                )
                rows = result.scalars().all()
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote listing failed.") from exc
        return [_quote_summary_from_row(row) for row in rows]

    async def list_user_quotes(self, user_id: str) -> list[QuoteSummaryRecord]:
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(QuoteRow)
                    .where(QuoteRow.user_id == user_id, QuoteRow.deleted_at_utc.is_(None))
                    .order_by(QuoteRow.created_at_utc.desc())
                )
                rows = result.scalars().all()
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote listing failed.") from exc
        return [_quote_summary_from_row(row) for row in rows]

    async def get_quote(
        self,
        *,
        quote_id: str,
        client_id_hash: str,
    ) -> StoredQuoteRecord | None:
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(QuoteRow).where(
                        QuoteRow.id == quote_id,
                        QuoteRow.client_id_hash == client_id_hash,
                        QuoteRow.deleted_at_utc.is_(None),
                    )
                )
                row = result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote retrieval failed.") from exc
        if row is None:
            return None
        return _stored_quote_from_row(row)

    async def get_user_quote(
        self,
        *,
        quote_id: str,
        user_id: str,
    ) -> StoredQuoteRecord | None:
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(QuoteRow).where(
                        QuoteRow.id == quote_id,
                        QuoteRow.user_id == user_id,
                        QuoteRow.deleted_at_utc.is_(None),
                    )
                )
                row = result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote retrieval failed.") from exc
        if row is None:
            return None
        return _stored_quote_from_row(row)

    async def get_any_quote(self, quote_id: str) -> StoredQuoteRecord | None:
        try:
            async with self.session() as session:
                row = await session.get(QuoteRow, quote_id)
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote retrieval failed.") from exc
        if row is None:
            return None
        return _stored_quote_from_row(row)

    async def list_admin_quotes(self, *, limit: int = 100) -> list[AdminQuoteSummaryRecord]:
        join_clause = outerjoin(QuoteRow, UserRow, QuoteRow.user_id == UserRow.id)
        try:
            async with self.session() as session:
                result = await session.execute(
                    select(QuoteRow, UserRow.email)
                    .select_from(join_clause)
                    .order_by(QuoteRow.created_at_utc.desc())
                    .limit(limit)
                )
                rows = result.all()
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote listing failed.") from exc
        return [_admin_quote_summary_from_joined_row(row[0], row[1]) for row in rows]

    async def delete_quote(self, quote_id: str) -> StoredQuoteRecord | None:
        try:
            async with self.session() as session, session.begin():
                row = await session.get(QuoteRow, quote_id)
                if row is None:
                    return None
                row.deleted_at_utc = datetime.now(UTC)
                await session.flush()
                await session.refresh(row)
        except SQLAlchemyError as exc:
            raise QuoteStoreUnavailableError("Quote deletion failed.") from exc
        return _stored_quote_from_row(row)

    async def _ensure_database_connection(self) -> None:
        if self._engine is None:
            raise AuditStoreUnavailableError("PostgreSQL audit store is not initialized.")
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise AuditStoreUnavailableError("Database connection check failed.") from exc

    async def _ensure_migrations_applied(self) -> None:
        if self._engine is None:
            raise AuditStoreUnavailableError("PostgreSQL audit store is not initialized.")

        config = _build_alembic_config(self.database_url)
        script_directory = ScriptDirectory.from_config(config)
        head_revision = script_directory.get_current_head()

        async with self._engine.connect() as connection:
            current_revision = await connection.run_sync(_get_current_revision)

        if current_revision != head_revision:
            raise AuditStoreUnavailableError(
                "Database schema is not at the latest Alembic revision."
            )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_windows_selector_event_loop_policy() -> None:
    if sys.platform != "win32" or not hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        return
    current_policy = asyncio.get_event_loop_policy()
    if current_policy.__class__.__name__ == "WindowsSelectorEventLoopPolicy":
        return
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _build_alembic_config(database_url: str) -> Config:
    config = Config(str(_repo_root() / "alembic.ini"))
    config.set_main_option("script_location", str(_repo_root() / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _get_current_revision(connection: Any) -> str | None:
    context = MigrationContext.configure(connection)
    return context.get_current_revision()


def _stored_user_from_row(row: UserRow) -> StoredUserRecord:
    return StoredUserRecord(
        id=row.id,
        created_at_utc=row.created_at_utc,
        email=row.email,
        role=row.role,  # type: ignore[arg-type]
        is_active=row.is_active,
        email_verified_at_utc=row.email_verified_at_utc,
    )


def _stored_auth_user_from_row(row: UserRow) -> StoredAuthUserRecord:
    return StoredAuthUserRecord(
        id=row.id,
        created_at_utc=row.created_at_utc,
        email=row.email,
        password_hash=row.password_hash,
        role=row.role,  # type: ignore[arg-type]
        is_active=row.is_active,
        email_verified_at_utc=row.email_verified_at_utc,
    )


def _admin_user_summary_from_row(row: UserRow) -> AdminUserSummaryRecord:
    return AdminUserSummaryRecord(
        id=row.id,
        created_at_utc=row.created_at_utc,
        email=row.email,
        role=row.role,  # type: ignore[arg-type]
        is_active=row.is_active,
        email_verified_at_utc=row.email_verified_at_utc,
    )


def _stored_quote_from_row(row: QuoteRow) -> StoredQuoteRecord:
    return StoredQuoteRecord(
        id=row.id,
        created_at_utc=row.created_at_utc,
        client_id_hash=row.client_id_hash,
        user_id=row.user_id,
        run_id=row.run_id,
        input_payload=dict(row.input_payload_json),
        frequency_prediction=row.frequency_prediction,
        severity_prediction=row.severity_prediction,
        prime_prediction=row.prime_prediction,
        deleted_at_utc=row.deleted_at_utc,
    )


def _quote_summary_from_row(row: QuoteRow) -> QuoteSummaryRecord:
    input_payload = dict(row.input_payload_json)
    return QuoteSummaryRecord(
        id=row.id,
        created_at_utc=row.created_at_utc,
        run_id=row.run_id,
        type_contrat=str(input_payload.get("type_contrat", "")),
        marque_vehicule=str(input_payload.get("marque_vehicule", "")),
        modele_vehicule=str(input_payload.get("modele_vehicule", "")),
        prime_prediction=row.prime_prediction,
    )


def _admin_quote_summary_from_joined_row(
    row: QuoteRow, owner_email: str | None
) -> AdminQuoteSummaryRecord:
    input_payload = dict(row.input_payload_json)
    return AdminQuoteSummaryRecord(
        id=row.id,
        created_at_utc=row.created_at_utc,
        run_id=row.run_id,
        type_contrat=str(input_payload.get("type_contrat", "")),
        marque_vehicule=str(input_payload.get("marque_vehicule", "")),
        modele_vehicule=str(input_payload.get("modele_vehicule", "")),
        prime_prediction=row.prime_prediction,
        user_id=row.user_id,
        owner_email=owner_email,
        deleted_at_utc=row.deleted_at_utc,
    )
