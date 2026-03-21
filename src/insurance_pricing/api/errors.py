from __future__ import annotations

import traceback

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from insurance_pricing.api.audit import (
    ApiErrorAuditRecord,
    AuditStoreUnavailableError,
    hash_raw_payload,
)
from insurance_pricing.api.logging import get_logger

ERROR_LOGGER = get_logger("insurance_pricing.api.error")


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown-request")
        payload_hash = await _request_payload_hash(request)
        ERROR_LOGGER.warning(
            "request_validation_failed",
            extra={
                "request_id": request_id,
                "status_code": 422,
                "endpoint": request.url.path,
                "errors": exc.errors(),
            },
        )
        await _persist_api_error_best_effort(
            request=request,
            record=ApiErrorAuditRecord(
                request_id=request_id,
                endpoint=request.url.path,
                run_id=getattr(request.app.state.settings, "run_id", None),
                status_code=422,
                exception_type="RequestValidationError",
                message="Request validation failed.",
                traceback_excerpt=None,
                payload_hash=payload_hash,
            ),
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(AuditStoreUnavailableError)
    async def handle_audit_store_unavailable(
        request: Request,
        exc: AuditStoreUnavailableError,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown-request")
        ERROR_LOGGER.error(
            "prediction_persistence_unavailable",
            extra={
                "request_id": request_id,
                "status_code": 503,
                "endpoint": request.url.path,
                "error": str(exc),
            },
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Prediction persistence is unavailable.",
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown-request")
        payload_hash = await _request_payload_hash(request)
        traceback_excerpt = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
            :4000
        ]
        ERROR_LOGGER.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "status_code": 500,
                "endpoint": request.url.path,
                "exception_type": type(exc).__name__,
            },
        )
        await _persist_api_error_best_effort(
            request=request,
            record=ApiErrorAuditRecord(
                request_id=request_id,
                endpoint=request.url.path,
                run_id=getattr(request.app.state.settings, "run_id", None),
                status_code=500,
                exception_type=type(exc).__name__,
                message=str(exc),
                traceback_excerpt=traceback_excerpt,
                payload_hash=payload_hash,
            ),
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "request_id": request_id,
            },
        )


async def _persist_api_error_best_effort(request: Request, record: ApiErrorAuditRecord) -> None:
    audit_store = getattr(request.app.state, "audit_store", None)
    if audit_store is None:
        return
    try:
        await audit_store.persist_api_error(record)
    except AuditStoreUnavailableError:
        ERROR_LOGGER.warning(
            "api_error_persistence_failed",
            extra={
                "request_id": record.request_id,
                "endpoint": record.endpoint,
                "status_code": record.status_code,
            },
        )


async def _request_payload_hash(request: Request) -> str | None:
    try:
        raw_body = await request.body()
    except RuntimeError:
        return None
    return hash_raw_payload(raw_body)
