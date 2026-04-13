from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from insurance_pricing.api import AppSettings, create_app
from insurance_pricing.api.dependencies import get_settings


def _register_verify_and_login(
    client: TestClient,
    *,
    client_id: str,
    email: str,
    password: str,
    in_memory_account_email_sender,
):
    register_response = client.post(
        "/auth/register",
        json={"email": email, "password": password},
        headers={"X-Client-ID": client_id},
    )
    verification_token = in_memory_account_email_sender.sent_emails[-1].verification_token
    verify_response = client.post("/auth/verify-email", json={"token": verification_token})
    login_response = client.post(
        "/auth/login",
        json={"email": email, "password": password},
        headers={"X-Client-ID": client_id},
    )
    return register_response, verify_response, login_response



def test_app_settings_require_run_id_and_database_url_without_env_sources(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("INSURANCE_PRICING_RUN_ID", raising=False)
    monkeypatch.delenv("INSURANCE_PRICING_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_invalid_run_id_fails_at_startup(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(
        api_settings.model_copy(update={"run_id": "run_does_not_exist"}),
        audit_store=in_memory_audit_store,
    )
    with pytest.raises(RuntimeError, match="run_does_not_exist"), TestClient(app):
        pass


def test_api_index_endpoint(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    with TestClient(app) as client:
        response = client.get("/")

    body = response.json()
    assert response.status_code == 200
    assert body["name"] == "Insurance Pricing API"
    assert body["docs_url"].endswith("/docs")
    assert body["redoc_url"].endswith("/redoc")
    assert body["openapi_url"].endswith("/openapi.json")
    assert body["health_url"].endswith("/health")
    assert body["ready_url"].endswith("/ready")
    assert body["version_url"].endswith("/version")
    assert body["current_model_url"].endswith("/models/current")
    assert body["prediction_schema_url"].endswith("/predict/schema")
    assert body["quotes_url"].endswith("/quotes")


def test_health_endpoint(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "run_id": api_settings.run_id,
        "model_loaded": True,
    }


def test_ready_endpoint_requires_database(
    api_settings: AppSettings,
    audit_store_factory,
):
    app = create_app(api_settings, audit_store=audit_store_factory(ready=False))

    with pytest.raises(RuntimeError, match="Failed to initialize"), TestClient(app):
        pass


def test_ready_endpoint_returns_200_when_store_ready(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["run_id"] == api_settings.run_id


def test_version_endpoint(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    with TestClient(app) as client:
        response = client.get("/version")

    body = response.json()
    assert response.status_code == 200
    assert body["api_version"]
    assert body["model_run_id"] == api_settings.run_id


def test_current_model_metadata_endpoint(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    with TestClient(app) as client:
        response = client.get("/models/current")

    body = response.json()
    assert response.status_code == 200
    assert body["run_id"] == api_settings.run_id
    assert set(body["model_files"]) == {"freq", "sev", "prime"}
    assert "feature_schema" in body
    assert "config" in body


def test_prediction_schema_endpoint(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    with TestClient(app) as client:
        response = client.get("/predict/schema")

    body = response.json()
    assert response.status_code == 200
    assert body["record_model"] == "InsurancePricingRecord"
    assert body["batch_model"] == "InsurancePricingBatchRequest"
    assert body["supports_batch"] is True
    assert "bonus" in body["required_fields"]
    assert body["optional_fields"] == []
    assert any(field["name"] == "bonus" for field in body["fields"])


def test_observability_headers_and_logs_are_added(
    api_settings: AppSettings,
    in_memory_audit_store,
    capsys: pytest.CaptureFixture[str],
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "test-request-id"})

    captured = capsys.readouterr().err
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0
    assert '"request_id": "test-request-id"' in captured
    assert f'"run_id": "{api_settings.run_id}"' in captured


@pytest.mark.parametrize(
    ("path", "expected_fields", "expected_kind"),
    [
        ("/predict/frequency", {"frequency_prediction"}, "frequency"),
        ("/predict/severity", {"severity_prediction"}, "severity"),
        (
            "/predict/prime",
            {"frequency_prediction", "severity_prediction", "prime_prediction"},
            "prime",
        ),
    ],
)
def test_predict_single_endpoints_persist_audit_record(
    api_settings: AppSettings,
    in_memory_audit_store,
    sample_prediction_records: list[dict],
    path: str,
    expected_fields: set[str],
    expected_kind: str,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)
    payload = sample_prediction_records[0]

    with TestClient(app) as client:
        response = client.post(path, json=payload, headers={"X-Request-ID": f"{expected_kind}-req"})

    body = response.json()
    assert response.status_code == 200
    assert set(body) == expected_fields
    assert len(in_memory_audit_store.predictions) == 1
    persisted = in_memory_audit_store.predictions[0]
    assert persisted.request_id == f"{expected_kind}-req"
    assert persisted.run_id == api_settings.run_id
    assert persisted.endpoint == path
    assert persisted.record_count == 1
    assert len(persisted.outputs) == 1
    assert persisted.outputs[0].record_position == 0


@pytest.mark.parametrize(
    ("path", "expected_prediction_keys", "prediction_method_name"),
    [
        ("/predict/frequency/batch", {"frequency_prediction"}, "predict_frequency_record"),
        ("/predict/severity/batch", {"severity_prediction"}, "predict_severity_record"),
        (
            "/predict/prime/batch",
            {"frequency_prediction", "severity_prediction", "prime_prediction"},
            "predict_record",
        ),
    ],
)
def test_predict_batch_endpoints_preserve_order_and_persist_outputs(
    api_settings: AppSettings,
    in_memory_audit_store,
    sample_prediction_records: list[dict],
    path: str,
    expected_prediction_keys: set[str],
    prediction_method_name: str,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)
    payload = {"records": list(reversed(sample_prediction_records))}

    with TestClient(app) as client:
        response = client.post(path, json=payload, headers={"X-Request-ID": "batch-request"})
        prediction_method = getattr(app.state.prediction_service, prediction_method_name)
        expected_predictions = [prediction_method(record) for record in payload["records"]]

    body = response.json()

    assert response.status_code == 200
    assert body["run_id"] == api_settings.run_id
    assert body["count"] == len(payload["records"])
    assert body["predictions"] == expected_predictions
    assert all(set(prediction) == expected_prediction_keys for prediction in body["predictions"])
    assert len(in_memory_audit_store.predictions) == 1
    persisted = in_memory_audit_store.predictions[0]
    assert persisted.request_id == "batch-request"
    assert len(persisted.outputs) == len(payload["records"])
    assert [output.record_position for output in persisted.outputs] == list(
        range(len(payload["records"]))
    )


@pytest.mark.parametrize(
    ("mutator", "expected_status"),
    [
        (lambda payload: payload.pop("bonus"), 422),
        (lambda payload: payload.__setitem__("bonus", "oops"), 422),
    ],
)
def test_predict_prime_rejects_invalid_payload_and_persists_error(
    api_settings: AppSettings,
    in_memory_audit_store,
    sample_prediction_records: list[dict],
    mutator,
    expected_status: int,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)
    payload = deepcopy(sample_prediction_records[0])
    mutator(payload)

    with TestClient(app) as client:
        response = client.post(
            "/predict/prime", json=payload, headers={"X-Request-ID": "invalid-req"}
        )

    assert response.status_code == expected_status
    assert response.json()["request_id"] == "invalid-req"
    assert len(in_memory_audit_store.errors) == 1
    assert in_memory_audit_store.errors[0].status_code == 422


def test_prediction_persistence_failure_returns_503(
    api_settings: AppSettings,
    sample_prediction_records: list[dict],
    audit_store_factory,
):
    app = create_app(
        api_settings,
        audit_store=audit_store_factory(fail_prediction_persistence=True),
    )

    with TestClient(app) as client:
        response = client.post("/predict/prime", json=sample_prediction_records[0])

    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction persistence is unavailable."


def test_create_quote_persists_payload_and_returns_prediction(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["run_id"] == api_settings.run_id
    assert "index" not in body["input_payload"]
    assert body["result"]["prime_prediction"] is not None
    assert body["email_delivery"] == {
        "status": "skipped",
        "recipient_email": "client@nova-assurances.fr",
        "detail": "Email delivery is not configured.",
    }
    assert len(in_memory_audit_store.quotes) == 1

def test_create_quote_requires_valid_client_id_header(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        missing_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={"X-Session-Token": session_token},
        )
        invalid_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": "not-a-uuid",
                "X-Session-Token": session_token,
            },
        )

    assert missing_response.status_code == 400
    assert missing_response.json()["detail"] == "X-Client-ID header is required."
    assert invalid_response.status_code == 400
    assert invalid_response.json()["detail"] == "X-Client-ID must be a valid UUID."

def test_quote_history_is_scoped_to_current_account(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_a = str(uuid4())
    client_b = str(uuid4())

    with TestClient(app) as client:
        _, _, login_a = _register_verify_and_login(
            client,
            client_id=client_a,
            email="client-a@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        _, _, login_b = _register_verify_and_login(
            client,
            client_id=client_b,
            email="client-b@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        token_a = login_a.json()["session_token"]
        token_b = login_b.json()["session_token"]
        first_quote = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_a,
                "X-Session-Token": token_a,
            },
        ).json()
        client.post(
            "/quotes",
            json=sample_prediction_records[1],
            headers={
                "X-Client-ID": client_b,
                "X-Session-Token": token_b,
            },
        )
        history_response = client.get("/quotes", headers={"X-Session-Token": token_a})
        own_quote_response = client.get(
            f"/quotes/{first_quote['id']}",
            headers={"X-Session-Token": token_a},
        )
        foreign_quote_response = client.get(
            f"/quotes/{first_quote['id']}",
            headers={"X-Session-Token": token_b},
        )

    history_body = history_response.json()
    assert history_response.status_code == 200
    assert history_body["count"] == 1
    assert history_body["quotes"][0]["id"] == first_quote["id"]
    assert own_quote_response.status_code == 200
    assert own_quote_response.json()["id"] == first_quote["id"]
    assert foreign_quote_response.status_code == 404
    assert foreign_quote_response.json()["detail"] == "Quote not found."

def test_quote_persistence_failure_returns_503(
    api_settings: AppSettings,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
    audit_store_factory,
):
    app = create_app(
        api_settings,
        audit_store=audit_store_factory(fail_quote_persistence=True),
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Quote persistence is unavailable."

def test_runtime_error_persists_api_error(
    api_settings: AppSettings,
    in_memory_audit_store,
    sample_prediction_records: list[dict],
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    def failing_predict_record(*args, **kwargs):
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        original_predict_record = app.state.prediction_service.predict_record
        app.state.prediction_service.predict_record = failing_predict_record
        response = client.post("/predict/prime", json=sample_prediction_records[0])
        app.state.prediction_service.predict_record = original_predict_record

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."
    assert len(in_memory_audit_store.errors) == 1
    assert in_memory_audit_store.errors[0].exception_type == "RuntimeError"


def test_openapi_exposes_product_endpoints(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)
    schema = app.openapi()

    assert schema["info"]["title"] == "Insurance Pricing API"
    assert (
        schema["info"]["summary"] == "HTTP inference API for the insurance premium pricing model."
    )
    assert "/predict/frequency" in schema["paths"]
    assert "/predict/severity" in schema["paths"]
    assert "/predict/prime" in schema["paths"]
    assert "/predict/schema" in schema["paths"]
    assert "/auth/register" in schema["paths"]
    assert "/auth/login" in schema["paths"]
    assert "/auth/session" in schema["paths"]
    assert "/auth/verify-email" in schema["paths"]
    assert "/auth/resend-verification-email" in schema["paths"]
    assert "/quotes" in schema["paths"]
    assert "/quotes/{quote_id}" in schema["paths"]
    assert "put" in schema["paths"]["/quotes/{quote_id}"]
    assert "/quotes/{quote_id}/report.pdf" in schema["paths"]
    assert "/quotes/{quote_id}/send-email" in schema["paths"]
    assert "/admin/users" in schema["paths"]
    assert "/admin/quotes" in schema["paths"]
    assert "/models/current" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert {tag["name"] for tag in schema["tags"]} == {
        "metadata",
        "health",
        "predict",
        "quotes",
        "auth",
        "admin",
    }
    assert schema["paths"]["/predict/prime"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]["description"]
    assert schema["paths"]["/predict/frequency"]["post"]["description"]


def test_swagger_ui_keeps_models_visible(
    api_settings: AppSettings,
    in_memory_audit_store,
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)

    assert app.swagger_ui_parameters["defaultModelsExpandDepth"] == 2
    assert app.swagger_ui_parameters["defaultModelExpandDepth"] == 2
    assert app.swagger_ui_parameters["defaultModelRendering"] == "model"


def test_register_requires_email_confirmation_before_quote_access(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        register_response = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": client_id},
        )
        session_response = client.get("/auth/session")
        blocked_history_response = client.get("/quotes")
        verification_token = in_memory_account_email_sender.sent_emails[0].verification_token
        verify_response = client.post("/auth/verify-email", json={"token": verification_token})
        login_response = client.post(
            "/auth/login",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": client_id},
        )
        session_token = login_response.json()["session_token"]
        history_response = client.get("/quotes", headers={"X-Session-Token": session_token})
        logout_response = client.post("/auth/logout", headers={"X-Session-Token": session_token})
        after_logout_response = client.get(
            "/auth/session",
            headers={"X-Session-Token": session_token},
        )

    assert register_response.status_code == 201
    assert register_response.json()["authenticated"] is False
    assert register_response.json()["session_token"] is None
    assert register_response.json()["email_verification_required"] is True
    assert session_response.status_code == 200
    assert session_response.json()["authenticated"] is False
    assert blocked_history_response.status_code == 401
    assert blocked_history_response.json()["detail"] == "Authentication is required."
    assert verify_response.status_code == 200
    assert login_response.status_code == 200
    assert login_response.json()["email_verification_required"] is False
    assert history_response.status_code == 200
    assert history_response.json()["count"] == 0
    assert logout_response.status_code == 204
    assert after_logout_response.status_code == 200
    assert after_logout_response.json()["authenticated"] is False

def test_unverified_user_must_confirm_email_before_logging_in(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )

    with TestClient(app) as client:
        register_response = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": str(uuid4())},
        )
        login_response = client.post(
            "/auth/login",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
        )

    assert register_response.status_code == 201
    assert login_response.status_code == 403
    assert login_response.json()["detail"] == "Veuillez confirmer votre adresse email avant de vous connecter."


def test_verify_email_endpoint_activates_future_logins(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        register_response = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": client_id},
        )
        verification_token = in_memory_account_email_sender.sent_emails[0].verification_token
        verify_response = client.post(
            "/auth/verify-email",
            json={"token": verification_token},
        )
        login_response = client.post(
            "/auth/login",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": client_id},
        )

    assert register_response.status_code == 201
    assert verify_response.status_code == 200
    assert verify_response.json()["email_verified_at_utc"] is not None
    assert login_response.status_code == 200
    assert login_response.json()["email_verification_required"] is False


def test_resend_verification_email_endpoint_reissues_a_new_email(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )

    with TestClient(app) as client:
        register_response = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": str(uuid4())},
        )
        first_token = in_memory_account_email_sender.sent_emails[0].verification_token
        resend_response = client.post(
            "/auth/resend-verification-email",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
        )

    assert register_response.status_code == 201
    assert resend_response.status_code == 200
    assert resend_response.json()["status"] == "sent"
    assert resend_response.json()["recipient_email"] == "client@nova-assurances.fr"
    assert len(in_memory_account_email_sender.sent_emails) == 2
    assert in_memory_account_email_sender.sent_emails[1].verification_token != first_token


def test_resend_verification_email_endpoint_skips_verified_accounts(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )

    with TestClient(app) as client:
        register_response = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": str(uuid4())},
        )
        verification_token = in_memory_account_email_sender.sent_emails[0].verification_token
        verify_response = client.post("/auth/verify-email", json={"token": verification_token})
        resend_response = client.post(
            "/auth/resend-verification-email",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
        )

    assert register_response.status_code == 201
    assert verify_response.status_code == 200
    assert resend_response.status_code == 200
    assert resend_response.json()["status"] == "skipped"
    assert resend_response.json()["detail"] == "This account is already verified."
    assert len(in_memory_account_email_sender.sent_emails) == 1


def test_resend_verification_email_endpoint_requires_valid_credentials(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )

    with TestClient(app) as client:
        register_response = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": str(uuid4())},
        )
        resend_response = client.post(
            "/auth/resend-verification-email",
            json={"email": "client@nova-assurances.fr", "password": "mauvais-mot-de-passe"},
        )

    assert register_response.status_code == 201
    assert resend_response.status_code == 401
    assert resend_response.json()["detail"] == "Invalid email or password."
    assert len(in_memory_account_email_sender.sent_emails) == 1


def test_authenticated_user_can_download_quote_report(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        quote_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )
        pdf_response = client.get(
            f"/quotes/{quote_response.json()['id']}/report.pdf",
            headers={"X-Session-Token": session_token},
        )

    assert quote_response.status_code == 200
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"] == "application/pdf"
    assert pdf_response.content.startswith(b"%PDF")

def test_authenticated_quote_creation_sends_recap_email_with_pdf(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    in_memory_quote_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
        quote_email_sender=in_memory_quote_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        quote_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )

    assert quote_response.status_code == 200
    assert quote_response.json()["email_delivery"] == {
        "status": "sent",
        "recipient_email": "client@nova-assurances.fr",
        "detail": "Email accepted by the in-memory sender.",
        "provider_status_code": 202,
    }
    assert len(in_memory_quote_email_sender.sent_emails) == 1
    sent_email = in_memory_quote_email_sender.sent_emails[0]
    assert sent_email.quote_id == quote_response.json()["id"]
    assert sent_email.recipient_email == "client@nova-assurances.fr"
    assert sent_email.pdf_bytes.startswith(b"%PDF")

def test_quote_creation_still_succeeds_when_email_delivery_fails(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    in_memory_quote_email_sender,
    sample_prediction_records: list[dict],
):
    in_memory_quote_email_sender.fail_send = True
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
        quote_email_sender=in_memory_quote_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        quote_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )

    assert quote_response.status_code == 200
    assert quote_response.json()["email_delivery"] == {
        "status": "failed",
        "recipient_email": "client@nova-assurances.fr",
        "detail": "Quote email delivery failed.",
    }
    assert in_memory_quote_email_sender.sent_emails == []

def test_updating_a_quote_reuses_the_same_quote_identifier(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())
    updated_payload = dict(sample_prediction_records[0])
    updated_payload["bonus"] = 0.77
    updated_payload["marque_vehicule"] = "PEUGEOT"

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        quote_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )
        update_response = client.put(
            f"/quotes/{quote_response.json()['id']}",
            json=updated_payload,
            headers={"X-Session-Token": session_token},
        )

    assert quote_response.status_code == 200
    assert update_response.status_code == 200
    assert update_response.json()["id"] == quote_response.json()["id"]
    assert update_response.json()["input_payload"]["bonus"] == 0.77
    assert update_response.json()["input_payload"]["marque_vehicule"] == "PEUGEOT"
    assert len(in_memory_audit_store.quotes) == 1

def test_authenticated_user_can_trigger_quote_email_endpoint(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    in_memory_quote_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
        quote_email_sender=in_memory_quote_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        quote_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )
        send_response = client.post(
            f"/quotes/{quote_response.json()['id']}/send-email",
            headers={"X-Session-Token": session_token},
        )

    assert send_response.status_code == 200
    assert send_response.json() == {
        "status": "sent",
        "recipient_email": "client@nova-assurances.fr",
        "detail": "Email accepted by the in-memory sender.",
        "provider_status_code": 202,
    }
    assert len(in_memory_quote_email_sender.sent_emails) == 2

def test_quote_email_endpoint_requires_authentication(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        _, _, login_response = _register_verify_and_login(
            client,
            client_id=client_id,
            email="client@nova-assurances.fr",
            password="motdepasse123",
            in_memory_account_email_sender=in_memory_account_email_sender,
        )
        session_token = login_response.json()["session_token"]
        quote_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )
        send_response = client.post(f"/quotes/{quote_response.json()['id']}/send-email")

    assert send_response.status_code == 401
    assert send_response.json()["detail"] == "Authentication is required."

def test_admin_can_list_and_moderate_users_and_quotes(
    api_settings: AppSettings,
    in_memory_audit_store,
    in_memory_account_email_sender,
    sample_prediction_records: list[dict],
):
    app = create_app(
        api_settings,
        audit_store=in_memory_audit_store,
        account_email_sender=in_memory_account_email_sender,
    )
    admin_client_id = str(uuid4())
    customer_client_id = str(uuid4())

    with TestClient(app) as client:
        customer_register = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": customer_client_id},
        )
        verify_customer = client.post(
            "/auth/verify-email",
            json={"token": in_memory_account_email_sender.sent_emails[0].verification_token},
        )
        customer_login = client.post(
            "/auth/login",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": customer_client_id},
        )
        admin_register = client.post(
            "/auth/register",
            json={"email": "admin@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": admin_client_id},
        )
        customer_token = customer_login.json()["session_token"]
        admin_token = admin_register.json()["session_token"]
        customer_quote = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={
                "X-Client-ID": customer_client_id,
                "X-Session-Token": customer_token,
            },
        )
        users_response = client.get("/admin/users", headers={"X-Session-Token": admin_token})
        quotes_response = client.get("/admin/quotes", headers={"X-Session-Token": admin_token})
        delete_quote_response = client.delete(
            f"/admin/quotes/{customer_quote.json()['id']}",
            headers={"X-Session-Token": admin_token},
        )
        deleted_quote_response = client.get(
            f"/quotes/{customer_quote.json()['id']}",
            headers={"X-Session-Token": customer_token},
        )
        delete_user_response = client.delete(
            f"/admin/users/{customer_register.json()['user']['id']}",
            headers={"X-Session-Token": admin_token},
        )

    assert customer_register.status_code == 201
    assert verify_customer.status_code == 200
    assert customer_login.status_code == 200
    assert admin_register.status_code == 201
    assert admin_register.json()["user"]["role"] == "admin"
    assert users_response.status_code == 200
    assert users_response.json()["count"] == 2
    assert quotes_response.status_code == 200
    assert quotes_response.json()["quotes"][0]["owner_email"] == "client@nova-assurances.fr"
    assert delete_quote_response.status_code == 204
    assert deleted_quote_response.status_code == 404
    assert delete_user_response.status_code == 204
