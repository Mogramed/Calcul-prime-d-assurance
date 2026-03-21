from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from insurance_pricing.api import AppSettings, create_app
from insurance_pricing.api.dependencies import get_settings


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
    assert "index" in body["optional_fields"]
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
        ("/predict/frequency", {"index", "frequency_prediction"}, "frequency"),
        ("/predict/severity", {"index", "severity_prediction"}, "severity"),
        (
            "/predict/prime",
            {"index", "frequency_prediction", "severity_prediction", "prime_prediction"},
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
    assert body["index"] == payload["index"]
    assert set(body) == expected_fields
    assert len(in_memory_audit_store.predictions) == 1
    persisted = in_memory_audit_store.predictions[0]
    assert persisted.request_id == f"{expected_kind}-req"
    assert persisted.run_id == api_settings.run_id
    assert persisted.endpoint == path
    assert persisted.record_count == 1
    assert len(persisted.outputs) == 1
    assert persisted.outputs[0].input_index == payload["index"]


@pytest.mark.parametrize(
    ("path", "expected_prediction_keys"),
    [
        ("/predict/frequency/batch", {"index", "frequency_prediction"}),
        ("/predict/severity/batch", {"index", "severity_prediction"}),
        (
            "/predict/prime/batch",
            {"index", "frequency_prediction", "severity_prediction", "prime_prediction"},
        ),
    ],
)
def test_predict_batch_endpoints_preserve_order_and_persist_outputs(
    api_settings: AppSettings,
    in_memory_audit_store,
    sample_prediction_records: list[dict],
    path: str,
    expected_prediction_keys: set[str],
):
    app = create_app(api_settings, audit_store=in_memory_audit_store)
    payload = {"records": list(reversed(sample_prediction_records))}

    with TestClient(app) as client:
        response = client.post(path, json=payload, headers={"X-Request-ID": "batch-request"})

    body = response.json()
    returned_indexes = [prediction["index"] for prediction in body["predictions"]]
    expected_indexes = [record["index"] for record in payload["records"]]

    assert response.status_code == 200
    assert body["run_id"] == api_settings.run_id
    assert body["count"] == len(payload["records"])
    assert returned_indexes == expected_indexes
    assert all(set(prediction) == expected_prediction_keys for prediction in body["predictions"])
    assert len(in_memory_audit_store.predictions) == 1
    persisted = in_memory_audit_store.predictions[0]
    assert persisted.request_id == "batch-request"
    assert len(persisted.outputs) == len(payload["records"])
    assert [output.input_index for output in persisted.outputs] == expected_indexes


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
    assert "/models/current" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert {tag["name"] for tag in schema["tags"]} == {"metadata", "health", "predict"}
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
