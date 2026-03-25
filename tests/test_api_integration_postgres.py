from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.rows import dict_row

from insurance_pricing.api import AppSettings, create_app

pytestmark = pytest.mark.integration

INTEGRATION_DATABASE_URL_ENV = "INSURANCE_PRICING_TEST_DATABASE_URL"
DEFAULT_INTEGRATION_DATABASE_URL = (
    "postgresql+psycopg://insurance_pricing:insurance_pricing@127.0.0.1:54329/insurance_pricing"
)


def _sync_dsn(database_url: str) -> str:
    return database_url.replace("+psycopg", "", 1)


def _replace_database_name(database_url: str, database_name: str) -> str:
    parsed = urlsplit(database_url)
    return urlunsplit(parsed._replace(path=f"/{database_name}"))


def _admin_dsn(database_url: str) -> str:
    return _sync_dsn(_replace_database_name(database_url, "postgres"))


@pytest.fixture(scope="session")
def integration_database_url() -> str:
    database_url = os.getenv(INTEGRATION_DATABASE_URL_ENV, DEFAULT_INTEGRATION_DATABASE_URL)
    try:
        with psycopg.connect(_sync_dsn(database_url), connect_timeout=2):
            pass
    except psycopg.OperationalError:
        pytest.skip(
            "PostgreSQL integration database is unavailable. Start docker compose first or set "
            f"{INTEGRATION_DATABASE_URL_ENV}."
        )
    return database_url


@pytest.fixture(autouse=True)
def clean_persistence_tables(integration_database_url: str):
    with (
        psycopg.connect(_sync_dsn(integration_database_url), autocommit=True) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(
            "TRUNCATE TABLE prediction_outputs, prediction_requests, "
            "api_errors, auth_sessions, users, quotes "
            "RESTART IDENTITY CASCADE"
        )
    yield


def _settings(existing_run_id: str, database_url: str) -> AppSettings:
    return AppSettings(
        run_id=existing_run_id,
        database_url=database_url,
        log_level="INFO",
        log_json=True,
    )


def test_ready_endpoint_with_postgres(existing_run_id: str, integration_database_url: str):
    app = create_app(_settings(existing_run_id, integration_database_url))

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["run_id"] == existing_run_id


def test_predict_prime_persists_postgres_rows(
    existing_run_id: str,
    integration_database_url: str,
    sample_prediction_records: list[dict],
):
    app = create_app(_settings(existing_run_id, integration_database_url))

    with TestClient(app) as client:
        response = client.post(
            "/predict/prime",
            json=sample_prediction_records[0],
            headers={"X-Request-ID": "postgres-prime-request"},
        )

    assert response.status_code == 200

    with psycopg.connect(_sync_dsn(integration_database_url), row_factory=dict_row) as connection:
        request_row = connection.execute(
            """
            SELECT request_id, endpoint, run_id, record_count, payload_hash, status_code
            FROM prediction_requests
            """
        ).fetchone()
        output_rows = connection.execute(
            """
            SELECT record_position, input_index, frequency_prediction, severity_prediction, prime_prediction
            FROM prediction_outputs
            ORDER BY record_position
            """
        ).fetchall()

    assert request_row is not None
    assert request_row["request_id"] == "postgres-prime-request"
    assert request_row["endpoint"] == "/predict/prime"
    assert request_row["run_id"] == existing_run_id
    assert request_row["record_count"] == 1
    assert request_row["payload_hash"]
    assert request_row["status_code"] == 200
    assert len(output_rows) == 1
    assert output_rows[0]["record_position"] == 0
    assert output_rows[0]["input_index"] == sample_prediction_records[0]["index"]
    assert output_rows[0]["prime_prediction"] is not None


def test_runtime_error_persists_api_error_row(
    existing_run_id: str,
    integration_database_url: str,
    sample_prediction_records: list[dict],
):
    app = create_app(_settings(existing_run_id, integration_database_url))

    def failing_predict_record(*args, **kwargs):
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        original_predict_record = app.state.prediction_service.predict_record
        app.state.prediction_service.predict_record = failing_predict_record
        response = client.post(
            "/predict/prime",
            json=sample_prediction_records[0],
            headers={"X-Request-ID": "postgres-error-request"},
        )
        app.state.prediction_service.predict_record = original_predict_record

    assert response.status_code == 500

    with psycopg.connect(_sync_dsn(integration_database_url), row_factory=dict_row) as connection:
        error_row = connection.execute(
            """
            SELECT request_id, endpoint, run_id, status_code, exception_type, message
            FROM api_errors
            """
        ).fetchone()

    assert error_row is not None
    assert error_row["request_id"] == "postgres-error-request"
    assert error_row["endpoint"] == "/predict/prime"
    assert error_row["run_id"] == existing_run_id
    assert error_row["status_code"] == 500
    assert error_row["exception_type"] == "RuntimeError"
    assert error_row["message"] == "boom"


def test_quotes_are_persisted_and_filtered_by_client_id(
    existing_run_id: str,
    integration_database_url: str,
    sample_prediction_records: list[dict],
):
    app = create_app(_settings(existing_run_id, integration_database_url))
    client_a = str(uuid4())
    client_b = str(uuid4())

    with TestClient(app) as client:
        first_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={"X-Client-ID": client_a},
        )
        second_response = client.post(
            "/quotes",
            json=sample_prediction_records[1],
            headers={"X-Client-ID": client_b},
        )
        history_response = client.get("/quotes", headers={"X-Client-ID": client_a})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert history_response.status_code == 200
    assert history_response.json()["count"] == 1
    assert history_response.json()["quotes"][0]["id"] == first_response.json()["id"]

    with psycopg.connect(_sync_dsn(integration_database_url), row_factory=dict_row) as connection:
        quote_rows = connection.execute(
            """
            SELECT client_id_hash, run_id, input_payload_json, prime_prediction
            FROM quotes
            ORDER BY created_at_utc
            """
        ).fetchall()

    assert len(quote_rows) == 2
    assert quote_rows[0]["run_id"] == existing_run_id
    assert quote_rows[0]["input_payload_json"]["index"] == sample_prediction_records[0]["index"]
    assert quote_rows[0]["prime_prediction"] is not None


def test_quote_detail_returns_404_for_foreign_client(
    existing_run_id: str,
    integration_database_url: str,
    sample_prediction_records: list[dict],
):
    app = create_app(_settings(existing_run_id, integration_database_url))
    client_a = str(uuid4())
    client_b = str(uuid4())

    with TestClient(app) as client:
        create_response = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={"X-Client-ID": client_a},
        )
        detail_response = client.get(
            f"/quotes/{create_response.json()['id']}",
            headers={"X-Client-ID": client_b},
        )

    assert create_response.status_code == 200
    assert detail_response.status_code == 404


def test_auth_registration_links_quotes_and_persists_users(
    existing_run_id: str,
    integration_database_url: str,
    sample_prediction_records: list[dict],
):
    app = create_app(
        _settings(
            existing_run_id,
            integration_database_url,
        )
    )
    client_id = str(uuid4())

    with TestClient(app) as client:
        anonymous_quote = client.post(
            "/quotes",
            json=sample_prediction_records[0],
            headers={"X-Client-ID": client_id},
        )
        register_response = client.post(
            "/auth/register",
            json={"email": "client@nova-assurances.fr", "password": "motdepasse123"},
            headers={"X-Client-ID": client_id},
        )
        session_token = register_response.json()["session_token"]
        history_response = client.get(
            "/quotes",
            headers={
                "X-Client-ID": client_id,
                "X-Session-Token": session_token,
            },
        )

    assert anonymous_quote.status_code == 200
    assert register_response.status_code == 201
    assert history_response.status_code == 200
    assert history_response.json()["count"] == 1

    with psycopg.connect(_sync_dsn(integration_database_url), row_factory=dict_row) as connection:
        user_row = connection.execute(
            """
            SELECT email, role, is_active
            FROM users
            """
        ).fetchone()
        quote_row = connection.execute(
            """
            SELECT user_id, deleted_at_utc
            FROM quotes
            """
        ).fetchone()

    assert user_row is not None
    assert user_row["email"] == "client@nova-assurances.fr"
    assert user_row["role"] == "customer"
    assert user_row["is_active"] is True
    assert quote_row is not None
    assert quote_row["user_id"] is not None
    assert quote_row["deleted_at_utc"] is None


def test_startup_fails_when_schema_is_not_migrated(
    existing_run_id: str,
    integration_database_url: str,
):
    isolated_database_name = f"insurance_pricing_unmigrated_{uuid4().hex[:8]}"
    unmigrated_database_url = _replace_database_name(
        integration_database_url, isolated_database_name
    )

    with psycopg.connect(_admin_dsn(integration_database_url), autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(isolated_database_name))
        )

    try:
        app = create_app(_settings(existing_run_id, unmigrated_database_url))
        with pytest.raises(RuntimeError, match="Failed to initialize"), TestClient(app):
            pass
    finally:
        with psycopg.connect(_admin_dsn(integration_database_url), autocommit=True) as connection:
            connection.execute(
                sql.SQL(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = {} AND pid <> pg_backend_pid()
                    """
                ).format(sql.Literal(isolated_database_name))
            )
            connection.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(isolated_database_name))
            )
