from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Response, status

from insurance_pricing.api.auth_store import (
    SessionCreateRecord,
    StoredAuthUserRecord,
    StoredUserRecord,
    UserAlreadyExistsError,
    UserCreateRecord,
    UserStore,
    UserStoreUnavailableError,
    hash_session_token,
    normalize_email,
)
from insurance_pricing.api.dependencies import (
    get_app_settings,
    get_current_user,
    get_optional_client_id,
    get_optional_session_token,
    get_user_store,
)
from insurance_pricing.api.quote_store import hash_client_id
from insurance_pricing.api.schemas import AuthCredentialsInput, AuthSessionResponse, UserResponse
from insurance_pricing.api.security import (
    generate_session_token,
    hash_password,
    session_expiry,
    verify_password,
)
from insurance_pricing.api.settings import AppSettings

router = APIRouter(tags=["auth"])


def _user_response(user: StoredUserRecord | StoredAuthUserRecord) -> UserResponse:
    return UserResponse(
        id=user.id,
        created_at_utc=user.created_at_utc,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
    )


def _session_response(
    user: StoredUserRecord | StoredAuthUserRecord | None,
    *,
    session_token: str | None = None,
    expires_at_utc: datetime | None = None,
) -> AuthSessionResponse:
    return AuthSessionResponse(
        authenticated=user is not None,
        user=_user_response(user) if user is not None else None,
        session_token=session_token,
        expires_at_utc=expires_at_utc,
    )


@router.post(
    "/auth/register",
    response_model=AuthSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a customer account",
    operation_id="register_account",
    responses={
        409: {"description": "A user already exists with this email."},
        503: {"description": "User persistence is unavailable."},
    },
)
async def register_account(
    payload: AuthCredentialsInput = Body(),
    client_id: str | None = Depends(get_optional_client_id),
    user_store: UserStore = Depends(get_user_store),
    settings: AppSettings = Depends(get_app_settings),
) -> AuthSessionResponse:
    email = normalize_email(str(payload.email))
    role: Literal["customer", "admin"] = (
        "admin" if email in settings.admin_emails else "customer"
    )

    try:
        user = await user_store.create_user(
            UserCreateRecord(
                email=email,
                password_hash=hash_password(payload.password),
                role=role,
            )
        )
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="An account already exists for this email.") from exc
    except UserStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Account creation is unavailable.") from exc

    session_token = generate_session_token()
    expires_at_utc = session_expiry(ttl_hours=settings.session_ttl_hours)

    try:
        await user_store.create_session(
            SessionCreateRecord(
                user_id=user.id,
                token_hash=hash_session_token(session_token),
                expires_at_utc=expires_at_utc,
            )
        )
        if client_id is not None:
            await user_store.attach_quotes_to_user(
                client_id_hash=hash_client_id(client_id),
                user_id=user.id,
            )
    except UserStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Account creation is unavailable.") from exc

    return _session_response(user, session_token=session_token, expires_at_utc=expires_at_utc)


@router.post(
    "/auth/login",
    response_model=AuthSessionResponse,
    summary="Authenticate a user account",
    operation_id="login_account",
    responses={
        401: {"description": "Invalid credentials."},
        503: {"description": "Authentication is unavailable."},
    },
)
async def login_account(
    payload: AuthCredentialsInput = Body(),
    client_id: str | None = Depends(get_optional_client_id),
    user_store: UserStore = Depends(get_user_store),
    settings: AppSettings = Depends(get_app_settings),
) -> AuthSessionResponse:
    email = normalize_email(str(payload.email))

    try:
        user = await user_store.get_user_auth_by_email(email)
    except UserStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Authentication is unavailable.") from exc

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account is inactive.")

    session_token = generate_session_token()
    expires_at_utc = session_expiry(ttl_hours=settings.session_ttl_hours)

    try:
        await user_store.create_session(
            SessionCreateRecord(
                user_id=user.id,
                token_hash=hash_session_token(session_token),
                expires_at_utc=expires_at_utc,
            )
        )
        if client_id is not None:
            await user_store.attach_quotes_to_user(
                client_id_hash=hash_client_id(client_id),
                user_id=user.id,
            )
    except UserStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Authentication is unavailable.") from exc

    return _session_response(user, session_token=session_token, expires_at_utc=expires_at_utc)


@router.get(
    "/auth/session",
    response_model=AuthSessionResponse,
    summary="Inspect the current authenticated session",
    operation_id="get_auth_session",
)
async def get_auth_session(
    current_user: StoredUserRecord | None = Depends(get_current_user),
) -> AuthSessionResponse:
    return _session_response(current_user)


@router.post(
    "/auth/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Close the current authenticated session",
    operation_id="logout_account",
)
async def logout_account(
    session_token: str | None = Depends(get_optional_session_token),
    user_store: UserStore = Depends(get_user_store),
) -> Response:
    if session_token is not None:
        try:
            await user_store.delete_session(hash_session_token(session_token))
        except UserStoreUnavailableError as exc:
            raise HTTPException(status_code=503, detail="Logout is unavailable.") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
