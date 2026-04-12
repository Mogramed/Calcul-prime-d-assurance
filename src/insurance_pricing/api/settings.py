from typing import Annotated

from pydantic import EmailStr, Field, SecretStr, field_validator, model_validator
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
    root_path: str = ""
    public_web_url: str | None = None
    resend_api_key: SecretStr | None = None
    resend_sender_email: EmailStr | None = None
    resend_sender_name: str = Field(default="Nova Assurances", min_length=1)

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

    @field_validator("root_path")
    @classmethod
    def normalize_root_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized == "/":
            return ""
        normalized = normalized.rstrip("/")
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    @field_validator("public_web_url")
    @classmethod
    def normalize_public_web_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return normalized.rstrip("/")

    @field_validator("resend_sender_name")
    @classmethod
    def normalize_resend_sender_name(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_resend_configuration(self) -> "AppSettings":
        if self.resend_api_key is not None and self.resend_sender_email is None:
            raise ValueError(
                "INSURANCE_PRICING_RESEND_SENDER_EMAIL is required when "
                "INSURANCE_PRICING_RESEND_API_KEY is configured."
            )
        return self


def _parse_list_setting(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError("Setting must be a string or a list of strings.")
