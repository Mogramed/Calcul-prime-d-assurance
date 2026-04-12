from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from insurance_pricing.api.logging import get_logger
from insurance_pricing.api.quote_store import StoredQuoteRecord

RESEND_SEND_EMAIL_URL = "https://api.resend.com/emails"
QUOTE_EMAIL_LOGGER = get_logger("insurance_pricing.api.quote_emailing")

_FIELD_LABELS: dict[str, str] = {
    "type_contrat": "Formule",
    "freq_paiement": "Paiement",
    "utilisation": "Usage",
    "age_conducteur1": "Conducteur principal",
    "conducteur2": "Second conducteur",
    "marque_vehicule": "Marque",
    "modele_vehicule": "Modele",
    "essence_vehicule": "Energie",
}

_FIELD_VALUE_LABELS: dict[str, dict[str, str]] = {
    "type_contrat": {
        "Mini": "Essentielle",
        "Median1": "Equilibre",
        "Median2": "Confort",
        "Maxi": "Premium",
    },
    "freq_paiement": {
        "Yearly": "Annuel",
        "Biannual": "Semestriel",
        "Quarterly": "Trimestriel",
        "Monthly": "Mensuel",
    },
    "utilisation": {
        "WorkPrivate": "Prive et domicile-travail",
        "Retired": "Retraite",
        "Professional": "Professionnel",
        "AllTrips": "Tous trajets",
    },
    "conducteur2": {
        "Yes": "Oui",
        "No": "Non",
    },
    "essence_vehicule": {
        "Gasoline": "Essence",
        "Diesel": "Diesel",
        "Electric": "Electrique",
        "Hybrid": "Hybride",
        "LPG": "GPL",
    },
}


@dataclass(frozen=True, slots=True)
class QuoteEmailDeliveryRecord:
    status: Literal["sent", "failed", "skipped"]
    recipient_email: str | None


class QuoteEmailSender(Protocol):
    async def send_quote_email(
        self,
        *,
        quote: StoredQuoteRecord,
        recipient_email: str,
        pdf_bytes: bytes,
    ) -> QuoteEmailDeliveryRecord: ...


class NoOpQuoteEmailSender:
    async def send_quote_email(
        self,
        *,
        quote: StoredQuoteRecord,
        recipient_email: str,
        pdf_bytes: bytes,
    ) -> QuoteEmailDeliveryRecord:
        return QuoteEmailDeliveryRecord(status="skipped", recipient_email=recipient_email)


class ResendQuoteEmailSender:
    def __init__(
        self,
        *,
        api_key: str,
        sender_email: str,
        sender_name: str,
        public_web_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.sender_email = sender_email
        self.sender_name = sender_name
        self.public_web_url = public_web_url.rstrip("/") if public_web_url else None

    async def send_quote_email(
        self,
        *,
        quote: StoredQuoteRecord,
        recipient_email: str,
        pdf_bytes: bytes,
    ) -> QuoteEmailDeliveryRecord:
        payload = {
            "from": f"{self.sender_name} <{self.sender_email}>",
            "to": [recipient_email],
            "subject": "Votre devis Nova Assurances",
            "html": _build_email_html(quote, self.public_web_url),
            "text": _build_email_text(quote, self.public_web_url),
            "attachments": [
                {
                    "filename": _quote_report_filename(quote),
                    "content": base64.b64encode(pdf_bytes).decode("ascii"),
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": f"quote-email:{quote.id}",
        }

        try:
            status_code, response_body = await asyncio.to_thread(
                _post_resend_email,
                payload,
                headers,
            )
        except OSError:
            QUOTE_EMAIL_LOGGER.warning(
                "quote_email_send_failed",
                extra={
                    "quote_id": quote.id,
                    "recipient_email": recipient_email,
                    "provider": "resend",
                },
                exc_info=True,
            )
            return QuoteEmailDeliveryRecord(status="failed", recipient_email=recipient_email)

        if 200 <= status_code < 300:
            provider_message_id = None
            try:
                provider_message_id = json.loads(response_body).get("id")
            except json.JSONDecodeError:
                provider_message_id = None
            QUOTE_EMAIL_LOGGER.info(
                "quote_email_sent",
                extra={
                    "quote_id": quote.id,
                    "recipient_email": recipient_email,
                    "provider": "resend",
                    "provider_message_id": provider_message_id,
                },
            )
            return QuoteEmailDeliveryRecord(status="sent", recipient_email=recipient_email)

        QUOTE_EMAIL_LOGGER.warning(
            "quote_email_send_rejected",
            extra={
                "quote_id": quote.id,
                "recipient_email": recipient_email,
                "provider": "resend",
                "status_code": status_code,
                "response_body": response_body[:500],
            },
        )
        return QuoteEmailDeliveryRecord(status="failed", recipient_email=recipient_email)


def build_quote_email_sender(
    *,
    resend_api_key: str | None,
    resend_sender_email: str | None,
    resend_sender_name: str,
    public_web_url: str | None,
) -> QuoteEmailSender:
    if not resend_api_key or not resend_sender_email:
        return NoOpQuoteEmailSender()
    return ResendQuoteEmailSender(
        api_key=resend_api_key,
        sender_email=resend_sender_email,
        sender_name=resend_sender_name,
        public_web_url=public_web_url,
    )


def _build_email_html(quote: StoredQuoteRecord, public_web_url: str | None) -> str:
    detail_url = _quote_detail_url(quote.id, public_web_url)
    summary_rows = "".join(
        f"""
        <tr>
          <td style="padding: 8px 0; color: #667085; font-size: 14px;">{escape(label)}</td>
          <td style="padding: 8px 0; color: #142132; font-size: 14px; font-weight: 600; text-align: right;">
            {escape(value)}
          </td>
        </tr>
        """
        for label, value in _summary_rows(quote)
    )
    cta_html = (
        f"""
        <p style="margin: 28px 0 0;">
          <a
            href="{escape(detail_url)}"
            style="display: inline-block; border-radius: 999px; background: #142132; color: #ffffff; padding: 12px 18px; text-decoration: none; font-weight: 600;"
          >
            Retrouver mon devis
          </a>
        </p>
        """
        if detail_url
        else ""
    )

    return f"""
    <div style="background: #f4efe9; padding: 32px; font-family: Arial, Helvetica, sans-serif; color: #142132;">
      <div style="max-width: 640px; margin: 0 auto; background: #ffffff; border-radius: 24px; padding: 32px; border: 1px solid #e8ddd0;">
        <p style="margin: 0 0 8px; font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: #b06d3c;">
          Nova Assurances
        </p>
        <h1 style="margin: 0; font-size: 30px; line-height: 1.1;">Votre devis auto est pret</h1>
        <p style="margin: 16px 0 0; font-size: 15px; line-height: 1.8; color: #475467;">
          Bonjour,<br/>
          Merci pour votre demande. Vous trouverez en piece jointe le PDF de votre devis, ainsi qu'un recapitulatif des informations principales ci-dessous.
        </p>

        <div style="margin-top: 24px; border-radius: 24px; padding: 24px; background: linear-gradient(180deg, #fffdfb, #f1e3d5); border: 1px solid #ead9c8;">
          <p style="margin: 0; font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: #b06d3c;">
            Prime estimee
          </p>
          <p style="margin: 12px 0 0; font-size: 42px; line-height: 1; font-weight: 700;">{escape(_format_currency(quote.prime_prediction))}</p>
          <p style="margin: 12px 0 0; font-size: 14px; color: #475467;">
            Reference {escape(quote.id)} - Emis le {escape(_format_datetime(quote.created_at_utc))}
          </p>
        </div>

        <table role="presentation" style="width: 100%; border-collapse: collapse; margin-top: 24px;">
          {summary_rows}
        </table>

        {cta_html}

        <p style="margin: 28px 0 0; font-size: 13px; line-height: 1.8; color: #667085;">
          Cette estimation est fournie a titre indicatif et sans engagement. Pour aller plus loin, vous pouvez revenir sur votre espace Nova Assurances a tout moment.
        </p>
      </div>
    </div>
    """


def _build_email_text(quote: StoredQuoteRecord, public_web_url: str | None) -> str:
    lines = [
        "Nova Assurances",
        "",
        "Votre devis auto est pret.",
        f"Prime estimee : {_format_currency(quote.prime_prediction)}",
        f"Reference : {quote.id}",
        f"Emis le : {_format_datetime(quote.created_at_utc)}",
        "",
        "Resume de votre demande :",
    ]
    lines.extend(f"- {label} : {value}" for label, value in _summary_rows(quote))
    detail_url = _quote_detail_url(quote.id, public_web_url)
    if detail_url:
        lines.extend(["", f"Retrouver mon devis : {detail_url}"])
    lines.extend(
        [
            "",
            "Le PDF de votre devis est joint a cet email.",
            "Cette estimation est fournie a titre indicatif et sans engagement.",
        ]
    )
    return "\n".join(lines)


def _summary_rows(quote: StoredQuoteRecord) -> list[tuple[str, str]]:
    payload = quote.input_payload
    rows = [
        (_FIELD_LABELS["type_contrat"], _friendly_field_value("type_contrat", payload.get("type_contrat"))),
        (_FIELD_LABELS["freq_paiement"], _friendly_field_value("freq_paiement", payload.get("freq_paiement"))),
        (_FIELD_LABELS["utilisation"], _friendly_field_value("utilisation", payload.get("utilisation"))),
        (
            _FIELD_LABELS["age_conducteur1"],
            f"{payload.get('age_conducteur1', 'Non renseigne')} ans",
        ),
        (_FIELD_LABELS["conducteur2"], _friendly_field_value("conducteur2", payload.get("conducteur2"))),
        (_FIELD_LABELS["marque_vehicule"], _friendly_field_value("marque_vehicule", payload.get("marque_vehicule"))),
        (_FIELD_LABELS["modele_vehicule"], _friendly_field_value("modele_vehicule", payload.get("modele_vehicule"))),
        (_FIELD_LABELS["essence_vehicule"], _friendly_field_value("essence_vehicule", payload.get("essence_vehicule"))),
    ]
    return rows


def _friendly_field_value(field_name: str, value: object) -> str:
    if value is None or value == "":
        return "Non renseigne"
    normalized = str(value)
    return _FIELD_VALUE_LABELS.get(field_name, {}).get(normalized, normalized.title())


def _quote_detail_url(quote_id: str, public_web_url: str | None) -> str | None:
    if not public_web_url:
        return None
    return f"{public_web_url}/mes-devis/{quote_id}"


def _quote_report_filename(quote: StoredQuoteRecord) -> str:
    return f"nova-devis-{quote.id}.pdf"


def _post_resend_email(
    payload: dict[str, object],
    headers: dict[str, str],
) -> tuple[int, str]:
    request = Request(
        RESEND_SEND_EMAIL_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            response_body = response.read().decode("utf-8")
            return response.status, response_body
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="ignore")
        return exc.code, response_body
    except URLError as exc:
        raise OSError("Unable to reach Resend.") from exc


def _format_currency(value: float) -> str:
    integer_part, decimal_part = f"{value:.2f}".split(".")
    grouped = f"{int(integer_part):,}".replace(",", " ")
    return f"{grouped},{decimal_part} EUR"


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%d/%m/%Y a %H:%M UTC")
