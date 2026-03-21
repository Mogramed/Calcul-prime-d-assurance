from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.exc import SQLAlchemyError
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
    input_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
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


class PostgresAuditStore(AuditStore):
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
                input_index=output.input_index,
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
