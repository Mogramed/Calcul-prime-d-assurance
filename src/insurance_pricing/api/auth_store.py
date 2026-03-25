from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Literal, Protocol

SESSION_TOKEN_HEADER = "X-Session-Token"
UserRole = Literal["customer", "admin"]


@dataclass(frozen=True, slots=True)
class UserCreateRecord:
    email: str
    password_hash: str
    role: UserRole


@dataclass(frozen=True, slots=True)
class StoredUserRecord:
    id: str
    created_at_utc: datetime
    email: str
    role: UserRole
    is_active: bool


@dataclass(frozen=True, slots=True)
class StoredAuthUserRecord:
    id: str
    created_at_utc: datetime
    email: str
    password_hash: str
    role: UserRole
    is_active: bool


@dataclass(frozen=True, slots=True)
class SessionCreateRecord:
    user_id: str
    token_hash: str
    expires_at_utc: datetime


@dataclass(frozen=True, slots=True)
class CreatedSessionRecord:
    session_token: str
    expires_at_utc: datetime


@dataclass(frozen=True, slots=True)
class AdminUserSummaryRecord:
    id: str
    created_at_utc: datetime
    email: str
    role: UserRole
    is_active: bool


class UserStoreUnavailableError(RuntimeError):
    """Raised when user persistence cannot complete."""


class UserAlreadyExistsError(RuntimeError):
    """Raised when a user with the same email already exists."""


class UserStore(Protocol):
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def check_ready(self) -> bool: ...

    async def create_user(self, record: UserCreateRecord) -> StoredUserRecord: ...

    async def get_user_auth_by_email(self, email: str) -> StoredAuthUserRecord | None: ...

    async def get_user_by_id(self, user_id: str) -> StoredUserRecord | None: ...

    async def get_user_by_session_token_hash(
        self, session_token_hash: str
    ) -> StoredUserRecord | None: ...

    async def create_session(self, record: SessionCreateRecord) -> None: ...

    async def delete_session(self, session_token_hash: str) -> None: ...

    async def list_admin_users(self) -> list[AdminUserSummaryRecord]: ...

    async def deactivate_user(self, user_id: str) -> StoredUserRecord | None: ...

    async def attach_quotes_to_user(self, *, client_id_hash: str, user_id: str) -> int: ...


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_session_token(session_token: str) -> str:
    return sha256(session_token.strip().encode("utf-8")).hexdigest()
