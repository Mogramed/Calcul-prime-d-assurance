from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Response, status

from insurance_pricing.api.account_emailing import AccountEmailDeliveryRecord, AccountEmailSender
from insurance_pricing.api.auth_store import (
    EmailVerificationCreateRecord,
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
    get_account_email_sender,
    get_app_settings,
    get_current_user,
    get_optional_client_id,
    get_optional_public_web_url,
    get_optional_session_token,
    get_user_store,
)
from insurance_pricing.api.logging import get_logger
from insurance_pricing.api.quote_store import hash_client_id
from insurance_pricing.api.schemas import (
    AuthCredentialsInput,
    AuthSessionResponse,
    EmailVerificationDeliveryResponse,
    EmailVerificationInput,
    EmailVerificationResendInput,
    UserResponse,
)
from insurance_pricing.api.security import (
    email_verification_expiry,
    generate_session_token,
    hash_password,
    session_expiry,
    verify_password,
)
from insurance_pricing.api.settings import AppSettings

router = APIRouter(tags=["auth"])
AUTH_ROUTER_LOGGER = get_logger("insurance_pricing.api.routers.auth")


def _user_response(user: StoredUserRecord | StoredAuthUserRecord) -> UserResponse:
    return UserResponse(
        id=user.id,
        created_at_utc=user.created_at_utc,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        email_verified_at_utc=user.email_verified_at_utc,
    )


def _email_delivery_response(
    delivery: AccountEmailDeliveryRecord | None,
) -> EmailVerificationDeliveryResponse | None:
    if delivery is None:
        return None
    return EmailVerificationDeliveryResponse(
        status=delivery.status,
        recipient_email=delivery.recipient_email,
        detail=delivery.detail,
        provider_status_code=delivery.provider_status_code,
    )


def _session_response(
    user: StoredUserRecord | StoredAuthUserRecord | None,
    *,
    session_token: str | None = None,
    expires_at_utc: datetime | None = None,
    email_verification_delivery: AccountEmailDeliveryRecord | None = None,
    authenticated: bool | None = None,
) -> AuthSessionResponse:
    return AuthSessionResponse(
        authenticated=user is not None if authenticated is None else authenticated,
        user=_user_response(user) if user is not None else None,
        session_token=session_token,
        expires_at_utc=expires_at_utc,
        email_verification_required=user is not None and user.email_verified_at_utc is None,
        email_verification_delivery=_email_delivery_response(email_verification_delivery),
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
    public_web_url: str | None = Depends(get_optional_public_web_url),
    user_store: UserStore = Depends(get_user_store),
    settings: AppSettings = Depends(get_app_settings),
    account_email_sender: AccountEmailSender = Depends(get_account_email_sender),
) -> AuthSessionResponse:
    email = normalize_email(str(payload.email))
    role: Literal["customer", "admin"] = (
        "admin" if email in settings.admin_emails else "customer"
    )
    email_verified_at_utc = datetime.now(UTC) if role == "admin" else None

    try:
        user = await user_store.create_user(
            UserCreateRecord(
                email=email,
                password_hash=hash_password(payload.password),
                role=role,
                email_verified_at_utc=email_verified_at_utc,
            )
        )
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="An account already exists for this email.") from exc
    except UserStoreUnavailableError as exc:
        AUTH_ROUTER_LOGGER.warning(
            "account_register_user_creation_failed",
            extra={
                "email": email,
                "role": role,
                "client_id_present": client_id is not None,
                "cause": repr(exc.__cause__) if exc.__cause__ is not None else str(exc),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail="Account creation is unavailable.") from exc

    session_token: str | None = None
    expires_at_utc: datetime | None = None
    email_verification_delivery: AccountEmailDeliveryRecord | None = None

    try:
        if client_id is not None:
            await user_store.attach_quotes_to_user(
                client_id_hash=hash_client_id(client_id),
                user_id=user.id,
            )
    except UserStoreUnavailableError as exc:
        AUTH_ROUTER_LOGGER.warning(
            "account_register_post_create_persistence_failed",
            extra={
                "email": email,
                "user_id": user.id,
                "client_id_present": client_id is not None,
                "email_verified": user.email_verified_at_utc is not None,
                "cause": repr(exc.__cause__) if exc.__cause__ is not None else str(exc),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=503, detail="Account creation is unavailable.") from exc

    if user.email_verified_at_utc is None:
        verification_token = generate_session_token()
        try:
            await user_store.create_email_verification(
                EmailVerificationCreateRecord(
                    user_id=user.id,
                    token_hash=hash_session_token(verification_token),
                    expires_at_utc=email_verification_expiry(),
                )
            )
        except UserStoreUnavailableError as exc:
            AUTH_ROUTER_LOGGER.warning(
                "account_register_email_token_creation_failed",
                extra={
                    "email": email,
                    "user_id": user.id,
                    "cause": repr(exc.__cause__) if exc.__cause__ is not None else str(exc),
                },
                exc_info=True,
            )
            raise HTTPException(status_code=503, detail="Account creation is unavailable.") from exc

        try:
            email_verification_delivery = await account_email_sender.send_verification_email(
                recipient_email=user.email,
                verification_token=verification_token,
                public_web_url=public_web_url or settings.public_web_url,
            )
        except Exception as exc:
            AUTH_ROUTER_LOGGER.warning(
                "account_register_verification_email_failed",
                extra={
                    "email": email,
                    "user_id": user.id,
                },
                exc_info=True,
            )
            email_verification_delivery = AccountEmailDeliveryRecord(
                status="failed",
                recipient_email=user.email,
                detail=str(exc),
            )
    else:
        try:
            session_token = generate_session_token()
            expires_at_utc = session_expiry(ttl_hours=settings.session_ttl_hours)
            await user_store.create_session(
                SessionCreateRecord(
                    user_id=user.id,
                    token_hash=hash_session_token(session_token),
                    expires_at_utc=expires_at_utc,
                )
            )
        except UserStoreUnavailableError as exc:
            AUTH_ROUTER_LOGGER.warning(
                "account_register_session_creation_failed",
                extra={
                    "email": email,
                    "user_id": user.id,
                    "cause": repr(exc.__cause__) if exc.__cause__ is not None else str(exc),
                },
                exc_info=True,
            )
            raise HTTPException(status_code=503, detail="Account creation is unavailable.") from exc

    return _session_response(
        user,
        session_token=session_token,
        expires_at_utc=expires_at_utc,
        email_verification_delivery=email_verification_delivery,
        authenticated=user.email_verified_at_utc is not None,
    )


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
    if user.email_verified_at_utc is None:
        raise HTTPException(
            status_code=403,
            detail="Veuillez confirmer votre adresse email avant de vous connecter.",
        )

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
    "/auth/verify-email",
    response_model=UserResponse,
    summary="Confirm a user email address",
    operation_id="verify_account_email",
    responses={
        400: {"description": "The verification token is missing or invalid."},
        404: {"description": "No pending verification matches this token."},
        503: {"description": "Email verification is unavailable."},
    },
)
async def verify_account_email(
    payload: EmailVerificationInput = Body(),
    user_store: UserStore = Depends(get_user_store),
) -> UserResponse:
    try:
        user = await user_store.verify_email(hash_session_token(payload.token))
    except UserStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Email verification is unavailable.") from exc

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="Ce lien de verification est invalide ou a expire.",
        )

    return _user_response(user)


@router.post(
    "/auth/resend-verification-email",
    response_model=EmailVerificationDeliveryResponse,
    summary="Resend the account verification email",
    operation_id="resend_account_verification_email",
    responses={
        401: {"description": "Invalid credentials."},
        403: {"description": "The account is inactive."},
        503: {"description": "Verification email delivery is unavailable."},
    },
)
async def resend_account_verification_email(
    payload: EmailVerificationResendInput = Body(),
    public_web_url: str | None = Depends(get_optional_public_web_url),
    user_store: UserStore = Depends(get_user_store),
    settings: AppSettings = Depends(get_app_settings),
    account_email_sender: AccountEmailSender = Depends(get_account_email_sender),
) -> EmailVerificationDeliveryResponse:
    email = normalize_email(str(payload.email))

    try:
        user = await user_store.get_user_auth_by_email(email)
    except UserStoreUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Verification email delivery is unavailable.",
        ) from exc

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account is inactive.")
    if user.email_verified_at_utc is not None:
        return EmailVerificationDeliveryResponse(
            status="skipped",
            recipient_email=user.email,
            detail="This account is already verified.",
        )

    verification_token = generate_session_token()
    try:
        await user_store.create_email_verification(
            EmailVerificationCreateRecord(
                user_id=user.id,
                token_hash=hash_session_token(verification_token),
                expires_at_utc=email_verification_expiry(),
            )
        )
    except UserStoreUnavailableError as exc:
        AUTH_ROUTER_LOGGER.warning(
            "account_resend_email_token_creation_failed",
            extra={
                "email": email,
                "user_id": user.id,
                "cause": repr(exc.__cause__) if exc.__cause__ is not None else str(exc),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Verification email delivery is unavailable.",
        ) from exc

    try:
        delivery = await account_email_sender.send_verification_email(
            recipient_email=user.email,
            verification_token=verification_token,
            public_web_url=public_web_url or settings.public_web_url,
        )
    except Exception as exc:
        AUTH_ROUTER_LOGGER.warning(
            "account_resend_verification_email_failed",
            extra={
                "email": email,
                "user_id": user.id,
            },
            exc_info=True,
        )
        delivery = AccountEmailDeliveryRecord(
            status="failed",
            recipient_email=user.email,
            detail=str(exc),
        )

    delivery_response = _email_delivery_response(delivery)
    if delivery_response is None:
        raise HTTPException(
            status_code=503,
            detail="Verification email delivery is unavailable.",
        )
    return delivery_response


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
