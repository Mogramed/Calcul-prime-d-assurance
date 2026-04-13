from __future__ import annotations

import asyncio

from insurance_pricing.api.account_emailing import (
    ResendAccountEmailSender,
    _build_account_verification_idempotency_key,
)


def test_resend_account_email_sender_sets_explicit_http_headers(monkeypatch):
    captured_headers: dict[str, str] = {}

    def fake_post_resend_email(payload, headers):
        del payload
        captured_headers.update(headers)
        return 202, '{"id":"email_123"}'

    monkeypatch.setattr(
        "insurance_pricing.api.account_emailing._post_resend_email",
        fake_post_resend_email,
    )

    sender = ResendAccountEmailSender(
        api_key="re_test",
        sender_email="contact@mohamed-khd.com",
        sender_name="Nova Assurances",
        public_web_url="https://nova-web.example.run.app",
    )

    result = asyncio.run(
        sender.send_verification_email(
            recipient_email="khaldimohamedamine78@gmail.com",
            verification_token="token_123",
        )
    )

    assert result.status == "sent"
    assert captured_headers["Authorization"] == "Bearer re_test"
    assert captured_headers["Accept"] == "application/json"
    assert captured_headers["Accept-Language"] == "fr-FR,fr;q=0.9,en;q=0.8"
    assert captured_headers["User-Agent"].startswith("NovaAssurances/")
    assert captured_headers["Idempotency-Key"] == _build_account_verification_idempotency_key(
        recipient_email="khaldimohamedamine78@gmail.com",
        verification_token="token_123",
    )


def test_account_verification_idempotency_key_changes_with_token():
    first_key = _build_account_verification_idempotency_key(
        recipient_email="client@nova-assurances.fr",
        verification_token="token_a",
    )
    second_key = _build_account_verification_idempotency_key(
        recipient_email="client@nova-assurances.fr",
        verification_token="token_b",
    )

    assert first_key.startswith("account-verification:")
    assert second_key.startswith("account-verification:")
    assert first_key != second_key
