from __future__ import annotations

import argparse
import secrets
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

SAMPLE_QUOTE_INPUT: dict[str, Any] = {
    "bonus": 0.58,
    "type_contrat": "Maxi",
    "duree_contrat": 1,
    "anciennete_info": 1,
    "freq_paiement": "Yearly",
    "paiement": "No",
    "utilisation": "Retired",
    "code_postal": 28388,
    "conducteur2": "No",
    "age_conducteur1": 66,
    "age_conducteur2": 0,
    "sex_conducteur1": "F",
    "sex_conducteur2": "",
    "anciennete_permis1": 34,
    "anciennete_permis2": 0,
    "anciennete_vehicule": 16.0,
    "cylindre_vehicule": 1239,
    "din_vehicule": 55,
    "essence_vehicule": "Gasoline",
    "marque_vehicule": "RENAULT",
    "modele_vehicule": "CLIO",
    "debut_vente_vehicule": 16,
    "fin_vente_vehicule": 15,
    "vitesse_vehicule": 150,
    "type_vehicule": "Tourism",
    "prix_vehicule": 10321,
    "poids_vehicule": 830,
}


@dataclass(slots=True)
class SmokeArtifacts:
    customer_email: str
    customer_user_id: str | None = None
    quote_id: str | None = None


def _log_ok(step: str, message: str) -> None:
    print(f"[ok] {step}: {message}")


def _fail(step: str, message: str) -> RuntimeError:
    return RuntimeError(f"[failed] {step}: {message}")


def _response_preview(response: httpx.Response) -> str:
    body = response.text.strip()
    if not body:
        return "<empty>"
    return body[:300]


def _request_path(path: str) -> str:
    normalized = path.lstrip("/")
    return normalized or "."


def _expect_status(
    response: httpx.Response,
    *,
    step: str,
    expected: int | tuple[int, ...],
) -> None:
    expected_values = (expected,) if isinstance(expected, int) else expected
    if response.status_code not in expected_values:
        raise _fail(
            step,
            f"expected status {expected_values}, got {response.status_code}. "
            f"Body: {_response_preview(response)}",
        )


def _check_page(client: httpx.Client, path: str, *, expected_text: str) -> None:
    step = f"page {path}"
    response = client.get(_request_path(path))
    _expect_status(response, step=step, expected=200)
    if expected_text not in response.text:
        raise _fail(step, f"expected to find '{expected_text}' in the HTML response.")
    _log_ok(step, "page rendered")


def _check_protected_page_redirect(
    client: httpx.Client,
    path: str,
    *,
    app_base_path: str,
    expected_redirect_path: str,
    expected_text: str,
) -> None:
    step = f"protected page {path}"
    response = client.get(_request_path(path))
    _expect_status(response, step=step, expected=200)
    final_expected_path = f"{app_base_path}{expected_redirect_path}" if app_base_path else expected_redirect_path
    if response.url.path != final_expected_path:
        raise _fail(
            step,
            f"expected final path '{final_expected_path}', got '{response.url.path}'.",
        )
    if expected_text not in response.text:
        raise _fail(step, f"expected to find '{expected_text}' in the redirected HTML response.")
    _log_ok(step, "authentication gate confirmed")


def _post_json(
    client: httpx.Client,
    path: str,
    *,
    step: str,
    payload: dict[str, Any] | None = None,
    expected: int | tuple[int, ...] = 200,
) -> dict[str, Any]:
    response = client.post(_request_path(path), json=payload)
    _expect_status(response, step=step, expected=expected)
    return response.json()


def _get_json(
    client: httpx.Client,
    path: str,
    *,
    step: str,
    expected: int | tuple[int, ...] = 200,
) -> dict[str, Any]:
    response = client.get(_request_path(path))
    _expect_status(response, step=step, expected=expected)
    return response.json()


def _delete(
    client: httpx.Client,
    path: str,
    *,
    step: str,
    expected: int | tuple[int, ...] = 204,
) -> None:
    response = client.delete(_request_path(path))
    _expect_status(response, step=step, expected=expected)


def _build_customer_email(prefix: str) -> str:
    timestamp = int(time.time())
    nonce = secrets.token_hex(3)
    return f"{prefix}-{timestamp}-{nonce}@example.com"


def _customer_flow(
    client: httpx.Client,
    *,
    email: str,
    password: str,
) -> SmokeArtifacts:
    register = _post_json(
        client,
        "/app-api/auth/register",
        step="customer registration",
        payload={"email": email, "password": password},
        expected=(200, 201),
    )
    if register.get("authenticated") is not True:
        raise _fail("customer registration", "the returned session is not authenticated.")

    user = register.get("user") or {}
    if user.get("email") != email:
        raise _fail("customer registration", "the returned user email does not match.")
    _log_ok("customer registration", email)

    session = _get_json(client, "/app-api/auth/session", step="session lookup")
    if session.get("authenticated") is not True:
        raise _fail("session lookup", "the session endpoint did not confirm the login.")
    _log_ok("session lookup", "authenticated session confirmed")

    quote = _post_json(
        client,
        "/app-api/quotes",
        step="quote creation",
        payload=SAMPLE_QUOTE_INPUT,
        expected=200,
    )
    quote_id = str(quote.get("id", ""))
    prime = quote.get("result", {}).get("prime_prediction")
    if not quote_id:
        raise _fail("quote creation", "missing quote identifier in response.")
    if not isinstance(prime, (int, float)) or prime <= 0:
        raise _fail("quote creation", "prime prediction was not returned as a positive number.")
    email_delivery = quote.get("email_delivery") or {}
    if email_delivery.get("status") not in {"sent", "failed", "skipped"}:
        raise _fail("quote creation", "email delivery status was not returned.")
    _log_ok("quote creation", f"quote {quote_id} created")

    history = _get_json(client, "/app-api/quotes", step="quote history")
    history_ids = {item.get("id") for item in history.get("quotes", [])}
    if quote_id not in history_ids:
        raise _fail("quote history", "the created quote was not found in history.")
    _log_ok("quote history", "quote visible in customer history")

    pdf_response = client.get(_request_path(f"/app-api/quotes/{quote_id}/report"))
    _expect_status(pdf_response, step="quote pdf", expected=200)
    content_type = pdf_response.headers.get("content-type", "")
    if "application/pdf" not in content_type:
        raise _fail("quote pdf", f"unexpected content type: {content_type!r}")
    if not pdf_response.content.startswith(b"%PDF"):
        raise _fail("quote pdf", "downloaded file does not look like a PDF.")
    _log_ok("quote pdf", "PDF report downloaded")

    return SmokeArtifacts(
        customer_email=email,
        customer_user_id=str(user.get("id")) if user.get("id") else None,
        quote_id=quote_id,
    )


def _admin_cleanup(
    *,
    base_url: str,
    timeout_seconds: float,
    admin_email: str,
    admin_password: str,
    admin_register_if_missing: bool,
    artifacts: SmokeArtifacts,
) -> None:
    with httpx.Client(
        base_url=base_url,
        follow_redirects=True,
        timeout=timeout_seconds,
        headers={"User-Agent": "nova-assurances-smoke/1.0"},
    ) as client:
        login_response = client.post(
            _request_path("/app-api/auth/login"),
            json={"email": admin_email, "password": admin_password},
        )
        if login_response.status_code == 401 and admin_register_if_missing:
            register_response = client.post(
                _request_path("/app-api/auth/register"),
                json={"email": admin_email, "password": admin_password},
            )
            _expect_status(
                register_response,
                step="admin bootstrap",
                expected=(200, 201, 409),
            )
            if register_response.status_code == 409:
                raise _fail(
                    "admin bootstrap",
                    "the admin account already exists but the provided password could not log in.",
                )
            register_body = register_response.json()
            role = (register_body.get("user") or {}).get("role")
            if role != "admin":
                raise _fail(
                    "admin bootstrap",
                    "the created account is not an admin. Add this email to "
                    "INSURANCE_PRICING_ADMIN_EMAILS before running the cleanup mode.",
                )
            _log_ok("admin bootstrap", admin_email)
        else:
            _expect_status(login_response, step="admin login", expected=200)
            login_body = login_response.json()
            role = (login_body.get("user") or {}).get("role")
            if role != "admin":
                raise _fail(
                    "admin login",
                    "the authenticated account is not an admin. Use an email listed in "
                    "INSURANCE_PRICING_ADMIN_EMAILS.",
                )
            _log_ok("admin login", admin_email)

        users = _get_json(client, "/app-api/admin/users", step="admin user list")
        quotes = _get_json(client, "/app-api/admin/quotes", step="admin quote list")
        if artifacts.customer_user_id and all(
            item.get("id") != artifacts.customer_user_id for item in users.get("users", [])
        ):
            raise _fail("admin user list", "the smoke user is missing from the admin view.")
        if artifacts.quote_id and all(
            item.get("id") != artifacts.quote_id for item in quotes.get("quotes", [])
        ):
            raise _fail("admin quote list", "the smoke quote is missing from the admin view.")
        _log_ok("admin views", "admin endpoints reachable")

        if artifacts.quote_id:
            _delete(client, f"/app-api/admin/quotes/{artifacts.quote_id}", step="admin quote cleanup")
            _log_ok("admin quote cleanup", artifacts.quote_id)
        if artifacts.customer_user_id:
            _delete(
                client,
                f"/app-api/admin/users/{artifacts.customer_user_id}",
                step="admin user cleanup",
            )
            _log_ok("admin user cleanup", artifacts.customer_email)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end smoke test against the public Nova Assurances web app.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Public base URL of the deployed Next.js service, for example https://nova-web-xyz.a.run.app",
    )
    parser.add_argument(
        "--customer-email-prefix",
        default="nova-smoke",
        help="Prefix used to generate the temporary customer email address.",
    )
    parser.add_argument(
        "--customer-email",
        help="Explicit customer email to use instead of a generated one.",
    )
    parser.add_argument(
        "--customer-password",
        default="NovaSmokePassword123!",
        help="Password used for the smoke customer account.",
    )
    parser.add_argument(
        "--admin-email",
        help="Optional admin email used to validate admin endpoints and clean up smoke data.",
    )
    parser.add_argument(
        "--admin-password",
        help="Password for the optional admin account.",
    )
    parser.add_argument(
        "--admin-register-if-missing",
        action="store_true",
        help=(
            "Attempt to register the admin account if login fails. The email must already be listed "
            "in INSURANCE_PRICING_ADMIN_EMAILS for the resulting account to be an admin."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout used for each request.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(args.admin_email) != bool(args.admin_password):
        print(
            "Both --admin-email and --admin-password must be provided together.",
            file=sys.stderr,
        )
        return 2

    customer_email = args.customer_email or _build_customer_email(args.customer_email_prefix)
    base_url = f"{args.base_url.rstrip('/')}/"
    app_base_path = urlsplit(base_url).path.rstrip("/")

    artifacts: SmokeArtifacts | None = None
    try:
        with httpx.Client(
            base_url=base_url,
            follow_redirects=True,
            timeout=args.timeout_seconds,
            headers={"User-Agent": "nova-assurances-smoke/1.0"},
        ) as client:
            _check_page(client, "/", expected_text="Nova Assurances")
            _check_protected_page_redirect(
                client,
                "/devis",
                app_base_path=app_base_path,
                expected_redirect_path="/connexion",
                expected_text="Connexion a votre espace",
            )
            artifacts = _customer_flow(
                client,
                email=customer_email,
                password=args.customer_password,
            )
            _check_page(client, "/devis", expected_text="Profil et formule")
            _check_page(client, "/mes-devis", expected_text="Retrouvez vos derniers devis")

        if artifacts and args.admin_email and args.admin_password:
            _admin_cleanup(
                base_url=base_url,
                timeout_seconds=args.timeout_seconds,
                admin_email=args.admin_email,
                admin_password=args.admin_password,
                admin_register_if_missing=args.admin_register_if_missing,
                artifacts=artifacts,
            )

        print()
        print("Smoke test completed successfully.")
        print(f"Base URL: {base_url}")
        print(f"Customer email: {customer_email}")
        if artifacts and artifacts.quote_id:
            print(f"Quote ID: {artifacts.quote_id}")
        if args.admin_email:
            print("Cleanup: completed through admin endpoints")
        else:
            print("Cleanup: not requested, the smoke user and quote remain in the database")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
