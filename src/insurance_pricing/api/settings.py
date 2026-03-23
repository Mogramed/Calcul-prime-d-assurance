from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INSURANCE_PRICING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    run_id: str = Field(min_length=1)
    database_url: str = Field(min_length=1)
    log_level: str = Field(default="INFO", min_length=1)
    log_json: bool = True
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    admin_emails: Annotated[list[str], NoDecode] = Field(default_factory=list)
    session_ttl_hours: int = Field(default=24 * 30, ge=1)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_allowed_origins(cls, value: object) -> list[str]:
        return _parse_list_setting(value)

    @field_validator("admin_emails", mode="before")
    @classmethod
    def parse_admin_emails(cls, value: object) -> list[str]:
        return [item.lower() for item in _parse_list_setting(value)]


def _parse_list_setting(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError("Setting must be a string or a list of strings.")
