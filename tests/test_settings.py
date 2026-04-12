import pytest
from pydantic import ValidationError

from insurance_pricing.api.settings import AppSettings


def test_app_settings_parses_cors_origins_from_csv_env(monkeypatch) -> None:
    monkeypatch.setenv("INSURANCE_PRICING_RUN_ID", "test-run")
    monkeypatch.setenv(
        "INSURANCE_PRICING_DATABASE_URL",
        "postgresql+psycopg://insurance_pricing:insurance_pricing@127.0.0.1:54329/insurance_pricing",
    )
    monkeypatch.setenv(
        "INSURANCE_PRICING_CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000",
    )

    settings = AppSettings()

    assert settings.cors_allowed_origins == [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]


def test_app_settings_parses_admin_emails_from_csv_env(monkeypatch) -> None:
    monkeypatch.setenv("INSURANCE_PRICING_RUN_ID", "test-run")
    monkeypatch.setenv(
        "INSURANCE_PRICING_DATABASE_URL",
        "postgresql+psycopg://insurance_pricing:insurance_pricing@127.0.0.1:54329/insurance_pricing",
    )
    monkeypatch.setenv(
        "INSURANCE_PRICING_ADMIN_EMAILS",
        "admin@nova-assurances.fr, second-admin@nova-assurances.fr ",
    )

    settings = AppSettings()

    assert settings.admin_emails == [
        "admin@nova-assurances.fr",
        "second-admin@nova-assurances.fr",
    ]


def test_app_settings_normalizes_root_path(monkeypatch) -> None:
    monkeypatch.setenv("INSURANCE_PRICING_RUN_ID", "test-run")
    monkeypatch.setenv(
        "INSURANCE_PRICING_DATABASE_URL",
        "postgresql+psycopg://insurance_pricing:insurance_pricing@127.0.0.1:54329/insurance_pricing",
    )
    monkeypatch.setenv("INSURANCE_PRICING_ROOT_PATH", "nova-assurance/api/")

    settings = AppSettings()

    assert settings.root_path == "/nova-assurance/api"


def test_app_settings_requires_sender_email_when_resend_is_configured(monkeypatch) -> None:
    monkeypatch.setenv("INSURANCE_PRICING_RUN_ID", "test-run")
    monkeypatch.setenv(
        "INSURANCE_PRICING_DATABASE_URL",
        "postgresql+psycopg://insurance_pricing:insurance_pricing@127.0.0.1:54329/insurance_pricing",
    )
    monkeypatch.setenv("INSURANCE_PRICING_RESEND_API_KEY", "re_test")
    monkeypatch.delenv("INSURANCE_PRICING_RESEND_SENDER_EMAIL", raising=False)

    with pytest.raises(ValidationError):
        AppSettings()
