from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

PASSWORD_HASHER = PasswordHasher()
DEFAULT_SESSION_TTL_HOURS = 24 * 30
DEFAULT_EMAIL_VERIFICATION_TTL_HOURS = 24


def hash_password(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return PASSWORD_HASHER.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)


def session_expiry(*, ttl_hours: int = DEFAULT_SESSION_TTL_HOURS) -> datetime:
    return datetime.now(UTC) + timedelta(hours=ttl_hours)


def email_verification_expiry(
    *, ttl_hours: int = DEFAULT_EMAIL_VERIFICATION_TTL_HOURS
) -> datetime:
    return datetime.now(UTC) + timedelta(hours=ttl_hours)
