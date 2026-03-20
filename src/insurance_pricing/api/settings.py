from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INSURANCE_PRICING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    run_id: str = Field(min_length=1)
