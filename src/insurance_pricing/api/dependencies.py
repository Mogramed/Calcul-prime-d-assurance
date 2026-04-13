from __future__ import annotations

from functools import lru_cache
from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request

from insurance_pricing.api.account_emailing import AccountEmailSender
from insurance_pricing.api.audit import AuditStore
from insurance_pricing.api.auth_store import (
    SESSION_TOKEN_HEADER,
    StoredUserRecord,
    UserStore,
    hash_session_token,
)
from insurance_pricing.api.quote_emailing import QuoteEmailSender
from insurance_pricing.api.quote_store import CLIENT_ID_HEADER, QuoteStore
from insurance_pricing.api.service import PredictionService
from insurance_pricing.api.settings import AppSettings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def get_prediction_service(request: Request) -> PredictionService:
    return cast(PredictionService, request.app.state.prediction_service)


def get_app_settings(request: Request) -> AppSettings:
    return cast(AppSettings, request.app.state.settings)


def get_audit_store(request: Request) -> AuditStore:
    return cast(AuditStore, request.app.state.audit_store)


def get_quote_store(request: Request) -> QuoteStore:
    return cast(QuoteStore, request.app.state.quote_store)


def get_user_store(request: Request) -> UserStore:
    return cast(UserStore, request.app.state.user_store)


def get_quote_email_sender(request: Request) -> QuoteEmailSender:
    return cast(QuoteEmailSender, request.app.state.quote_email_sender)


def get_account_email_sender(request: Request) -> AccountEmailSender:
    return cast(AccountEmailSender, request.app.state.account_email_sender)


def get_client_id(
    client_id: Annotated[str | None, Header(alias=CLIENT_ID_HEADER)] = None,
) -> str:
    if client_id is None:
        raise HTTPException(status_code=400, detail=f"{CLIENT_ID_HEADER} header is required.")
    try:
        return str(UUID(client_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{CLIENT_ID_HEADER} must be a valid UUID.",
        ) from exc


def get_optional_client_id(
    client_id: Annotated[str | None, Header(alias=CLIENT_ID_HEADER)] = None,
) -> str | None:
    if client_id is None:
        return None
    try:
        return str(UUID(client_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{CLIENT_ID_HEADER} must be a valid UUID.",
        ) from exc


def get_optional_session_token(
    session_token: Annotated[str | None, Header(alias=SESSION_TOKEN_HEADER)] = None,
) -> str | None:
    if session_token is None:
        return None
    normalized = session_token.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{SESSION_TOKEN_HEADER} must not be empty.")
    return normalized


async def get_current_user(
    user_store: UserStore = Depends(get_user_store),
    session_token: str | None = Depends(get_optional_session_token),
) -> StoredUserRecord | None:
    if session_token is None:
        return None
    user = await user_store.get_user_by_session_token_hash(hash_session_token(session_token))
    if user is None:
        return None
    if not user.is_active or user.email_verified_at_utc is None:
        return None
    return user


async def get_authenticated_user(
    current_user: StoredUserRecord | None = Depends(get_current_user),
) -> StoredUserRecord:
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    return current_user


async def get_admin_user(
    current_user: StoredUserRecord = Depends(get_authenticated_user),
) -> StoredUserRecord:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access is required.")
    return current_user
