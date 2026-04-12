from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from insurance_pricing.api.quote_emailing import ResendQuoteEmailSender
from insurance_pricing.api.quote_store import StoredQuoteRecord


def test_resend_quote_email_sender_sets_explicit_http_headers(monkeypatch):
    captured_headers: dict[str, str] = {}

    def fake_post_resend_email(payload, headers):
        del payload
        captured_headers.update(headers)
        return 202, '{"id":"email_123"}'

    monkeypatch.setattr(
        "insurance_pricing.api.quote_emailing._post_resend_email",
        fake_post_resend_email,
    )

    sender = ResendQuoteEmailSender(
        api_key="re_test",
        sender_email="contact@mohamed-khd.com",
        sender_name="Nova Assurances",
        public_web_url="https://nova-web.example.run.app",
    )

    quote = StoredQuoteRecord(
        id="quote_123",
        created_at_utc=datetime.now(UTC),
        client_id_hash="client_hash",
        user_id="user_123",
        run_id="run_123",
        input_payload={
            "type_contrat": "Maxi",
            "freq_paiement": "Yearly",
            "utilisation": "Retired",
            "age_conducteur1": 55,
            "conducteur2": "No",
            "marque_vehicule": "PEUGEOT",
            "modele_vehicule": "307",
            "essence_vehicule": "Gasoline",
        },
        frequency_prediction=0.1,
        severity_prediction=1000.0,
        prime_prediction=100.0,
        deleted_at_utc=None,
    )

    result = asyncio.run(
        sender.send_quote_email(
            quote=quote,
            recipient_email="khaldimohamedamine78@gmail.com",
            pdf_bytes=b"%PDF-1.7\n",
        )
    )

    assert result.status == "sent"
    assert captured_headers["Authorization"] == "Bearer re_test"
    assert captured_headers["Accept"] == "application/json"
    assert captured_headers["Accept-Language"] == "fr-FR,fr;q=0.9,en;q=0.8"
    assert captured_headers["User-Agent"].startswith("NovaAssurances/")
    assert captured_headers["Idempotency-Key"] == "quote-email:quote_123"
