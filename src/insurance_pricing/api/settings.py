from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()
