from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from insurance_pricing import __version__
from insurance_pricing.api.logging import get_logger

RESEND_SEND_EMAIL_URL = "https://api.resend.com/emails"
ACCOUNT_EMAIL_LOGGER = get_logger("insurance_pricing.api.account_emailing")


@dataclass(frozen=True, slots=True)
class AccountEmailDeliveryRecord:
    status: Literal["sent", "failed", "skipped"]
    recipient_email: str | None
    detail: str | None = None
    provider_status_code: int | None = None


class AccountEmailSender(Protocol):
    async def send_verification_email(
        self,
        *,
        recipient_email: str,
        verification_token: str,
        public_web_url: str | None = None,
    ) -> AccountEmailDeliveryRecord: ...


class NoOpAccountEmailSender:
    async def send_verification_email(
        self,
        *,
        recipient_email: str,
        verification_token: str,
        public_web_url: str | None = None,
    ) -> AccountEmailDeliveryRecord:
        del verification_token, public_web_url
        return AccountEmailDeliveryRecord(
            status="skipped",
            recipient_email=recipient_email,
            detail="Email verification delivery is not configured.",
        )


class ResendAccountEmailSender:
    def __init__(
        self,
        *,
        api_key: str,
        sender_email: str,
        sender_name: str,
        public_web_url: str | None,
    ) -> None:
        self.api_key = api_key
        self.sender_email = sender_email
        self.sender_name = sender_name
        self.public_web_url = public_web_url.rstrip("/") if public_web_url else None

    async def send_verification_email(
        self,
        *,
        recipient_email: str,
        verification_token: str,
        public_web_url: str | None = None,
    ) -> AccountEmailDeliveryRecord:
        resolved_public_web_url = (public_web_url or self.public_web_url or "").strip().rstrip("/")
        if not resolved_public_web_url:
            ACCOUNT_EMAIL_LOGGER.warning(
                "account_verification_email_skipped",
                extra={
                    "recipient_email": recipient_email,
                    "provider": "resend",
                    "reason": "missing_public_web_url",
                },
            )
            return AccountEmailDeliveryRecord(
                status="skipped",
                recipient_email=recipient_email,
                detail="Public web URL is not configured for email verification.",
            )

        verification_url = (
            f"{resolved_public_web_url}/verification-email?token={verification_token}"
        )
        payload: Mapping[str, object] = {
            "from": f"{self.sender_name} <{self.sender_email}>",
            "to": [recipient_email],
            "subject": "Confirmez votre adresse email Nova Assurances",
            "html": _build_verification_email_html(verification_url),
            "text": _build_verification_email_text(verification_url),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Content-Type": "application/json",
            "Idempotency-Key": f"account-verification:{recipient_email.lower()}",
            "User-Agent": f"NovaAssurances/{__version__} (Cloud Run; Resend API client)",
        }

        ACCOUNT_EMAIL_LOGGER.info(
            "account_verification_email_requested",
            extra={"recipient_email": recipient_email, "provider": "resend"},
        )

        try:
            status_code, response_body = await asyncio.to_thread(
                _post_resend_email,
                payload,
                headers,
            )
        except OSError:
            ACCOUNT_EMAIL_LOGGER.warning(
                "account_verification_email_failed",
                extra={"recipient_email": recipient_email, "provider": "resend"},
                exc_info=True,
            )
            return AccountEmailDeliveryRecord(
                status="failed",
                recipient_email=recipient_email,
                detail="Unable to reach Resend.",
            )

        if 200 <= status_code < 300:
            ACCOUNT_EMAIL_LOGGER.info(
                "account_verification_email_sent",
                extra={"recipient_email": recipient_email, "provider": "resend"},
            )
            return AccountEmailDeliveryRecord(
                status="sent",
                recipient_email=recipient_email,
                detail="Verification email accepted by Resend.",
                provider_status_code=status_code,
            )

        error_detail = _extract_resend_error_detail(response_body)
        ACCOUNT_EMAIL_LOGGER.warning(
            "account_verification_email_rejected",
            extra={
                "recipient_email": recipient_email,
                "provider": "resend",
                "status_code": status_code,
                "response_body": response_body[:500],
                "provider_detail": error_detail,
            },
        )
        return AccountEmailDeliveryRecord(
            status="failed",
            recipient_email=recipient_email,
            detail=error_detail,
            provider_status_code=status_code,
        )


def build_account_email_sender(
    *,
    resend_api_key: str | None,
    resend_sender_email: str | None,
    resend_sender_name: str,
    public_web_url: str | None,
) -> AccountEmailSender:
    if not resend_api_key or not resend_sender_email:
        return NoOpAccountEmailSender()
    return ResendAccountEmailSender(
        api_key=resend_api_key,
        sender_email=resend_sender_email,
        sender_name=resend_sender_name,
        public_web_url=public_web_url,
    )


def _build_verification_email_html(verification_url: str) -> str:
    return f"""
    <div style="background: #f4efe9; padding: 32px; font-family: Arial, Helvetica, sans-serif; color: #142132;">
      <div style="max-width: 640px; margin: 0 auto; background: #ffffff; border-radius: 24px; padding: 32px; border: 1px solid #e8ddd0;">
        <p style="margin: 0 0 8px; font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase; color: #b06d3c;">
          Nova Assurances
        </p>
        <h1 style="margin: 0; font-size: 30px; line-height: 1.1;">Confirmez votre adresse email</h1>
        <p style="margin: 16px 0 0; font-size: 15px; line-height: 1.8; color: #475467;">
          Merci pour votre inscription. Cliquez sur le bouton ci-dessous pour valider votre compte et
          conserver l'acces a votre espace client.
        </p>

        <p style="margin: 28px 0 0;">
          <a
            href="{escape(verification_url)}"
            style="display: inline-block; border-radius: 999px; background: #142132; color: #ffffff; padding: 12px 18px; text-decoration: none; font-weight: 600;"
          >
            Confirmer mon adresse email
          </a>
        </p>

        <p style="margin: 24px 0 0; font-size: 13px; line-height: 1.8; color: #667085;">
          Si le bouton ne fonctionne pas, copiez ce lien dans votre navigateur :
          <br />
          <a href="{escape(verification_url)}" style="color: #b06d3c;">{escape(verification_url)}</a>
        </p>
      </div>
    </div>
    """


def _build_verification_email_text(verification_url: str) -> str:
    return "\n".join(
        [
            "Nova Assurances",
            "",
            "Merci pour votre inscription.",
            "Confirmez votre adresse email en ouvrant le lien ci-dessous :",
            verification_url,
        ]
    )


def _post_resend_email(
    payload: Mapping[str, object],
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
            body = response.read().decode("utf-8")
            return response.status, body
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except URLError as exc:
        raise OSError("Unable to reach Resend.") from exc


def _extract_resend_error_detail(response_body: str) -> str:
    if not response_body:
        return "Resend rejected the request."
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return response_body[:200]
    if isinstance(parsed, dict):
        message = parsed.get("message")
        if isinstance(message, str) and message.strip():
            return message
        error = parsed.get("error")
        if isinstance(error, str) and error.strip():
            return error
        name = parsed.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return "Resend rejected the request."
