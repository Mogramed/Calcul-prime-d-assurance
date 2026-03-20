from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from insurance_pricing.api import AppSettings, create_app
from insurance_pricing.api.dependencies import get_settings


def test_app_settings_require_run_id_without_env_sources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INSURANCE_PRICING_RUN_ID", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_invalid_run_id_fails_at_startup():
    app = create_app(AppSettings(run_id="run_does_not_exist"))
    with pytest.raises(RuntimeError, match="run_does_not_exist"):
        with TestClient(app):
            pass


def test_api_index_endpoint(existing_run_id: str):
    app = create_app(AppSettings(run_id=existing_run_id))

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


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_health_endpoints(existing_run_id: str, path: str):
    app = create_app(AppSettings(run_id=existing_run_id))

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "run_id": existing_run_id,
        "model_loaded": True,
    }


def test_version_endpoint(existing_run_id: str):
    app = create_app(AppSettings(run_id=existing_run_id))

    with TestClient(app) as client:
        response = client.get("/version")

    body = response.json()
    assert response.status_code == 200
    assert body["api_version"]
    assert body["model_run_id"] == existing_run_id


def test_current_model_metadata_endpoint(existing_run_id: str):
    app = create_app(AppSettings(run_id=existing_run_id))

    with TestClient(app) as client:
        response = client.get("/models/current")

    body = response.json()
    assert response.status_code == 200
    assert body["run_id"] == existing_run_id
    assert set(body["model_files"]) == {"freq", "sev", "prime"}
    assert "feature_schema" in body
    assert "config" in body


def test_prediction_schema_endpoint(existing_run_id: str):
    app = create_app(AppSettings(run_id=existing_run_id))

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


def test_observability_headers_are_added(existing_run_id: str):
    app = create_app(AppSettings(run_id=existing_run_id))

    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0


@pytest.mark.parametrize(
    ("path", "expected_fields"),
    [
        ("/predict/frequency", {"index", "frequency_prediction"}),
        ("/predict/severity", {"index", "severity_prediction"}),
        (
            "/predict/prime",
            {"index", "frequency_prediction", "severity_prediction", "prime_prediction"},
        ),
    ],
)
def test_predict_single_endpoints(
    existing_run_id: str,
    sample_prediction_records: list[dict],
    path: str,
    expected_fields: set[str],
):
    app = create_app(AppSettings(run_id=existing_run_id))
    payload = sample_prediction_records[0]

    with TestClient(app) as client:
        response = client.post(path, json=payload)

    body = response.json()
    assert response.status_code == 200
    assert body["index"] == payload["index"]
    assert set(body) == expected_fields
    for field in expected_fields - {"index"}:
        assert isinstance(body[field], float)


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
def test_predict_batch_endpoints_preserve_order(
    existing_run_id: str,
    sample_prediction_records: list[dict],
    path: str,
    expected_prediction_keys: set[str],
):
    app = create_app(AppSettings(run_id=existing_run_id))
    payload = {"records": list(reversed(sample_prediction_records))}

    with TestClient(app) as client:
        response = client.post(path, json=payload)

    body = response.json()
    returned_indexes = [prediction["index"] for prediction in body["predictions"]]
    expected_indexes = [record["index"] for record in payload["records"]]

    assert response.status_code == 200
    assert body["run_id"] == existing_run_id
    assert body["count"] == len(payload["records"])
    assert returned_indexes == expected_indexes
    assert all(set(prediction) == expected_prediction_keys for prediction in body["predictions"])


@pytest.mark.parametrize(
    ("mutator", "expected_status"),
    [
        (lambda payload: payload.pop("bonus"), 422),
        (lambda payload: payload.__setitem__("bonus", "oops"), 422),
    ],
)
def test_predict_prime_rejects_invalid_payload(
    existing_run_id: str,
    sample_prediction_records: list[dict],
    mutator,
    expected_status: int,
):
    app = create_app(AppSettings(run_id=existing_run_id))
    payload = deepcopy(sample_prediction_records[0])
    mutator(payload)

    with TestClient(app) as client:
        response = client.post("/predict/prime", json=payload)

    assert response.status_code == expected_status


def test_openapi_exposes_product_endpoints(existing_run_id: str):
    app = create_app(AppSettings(run_id=existing_run_id))
    schema = app.openapi()

    assert schema["info"]["title"] == "Insurance Pricing API"
    assert schema["info"]["summary"] == "HTTP inference API for the insurance premium pricing model."
    assert "/predict/frequency" in schema["paths"]
    assert "/predict/severity" in schema["paths"]
    assert "/predict/prime" in schema["paths"]
    assert "/predict/schema" in schema["paths"]
    assert "/models/current" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert {tag["name"] for tag in schema["tags"]} == {"metadata", "health", "predict"}
    assert (
        schema["paths"]["/predict/prime"]["post"]["requestBody"]["content"]["application/json"][
            "schema"
        ]["description"]
    )
    assert schema["paths"]["/predict/frequency"]["post"]["description"]


def test_swagger_ui_keeps_models_visible(existing_run_id: str):
    app = create_app(AppSettings(run_id=existing_run_id))

    assert app.swagger_ui_parameters["defaultModelsExpandDepth"] == 2
    assert app.swagger_ui_parameters["defaultModelExpandDepth"] == 2
    assert app.swagger_ui_parameters["defaultModelRendering"] == "model"
