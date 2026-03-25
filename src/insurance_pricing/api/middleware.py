from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.responses import Response

from insurance_pricing.api.logging import (
    bind_request_context,
    clear_request_context,
    get_logger,
)

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"
REQUEST_LOGGER = get_logger("insurance_pricing.api.request")


def install_observability_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_observability(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
        request.state.request_id = request_id
        started_at = perf_counter()
        request.state.started_at = started_at
        bind_request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            run_id=getattr(request.app.state.settings, "run_id", None),
        )

        try:
            response = await call_next(request)
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.2f}"
            REQUEST_LOGGER.info(
                "request_completed",
                extra={
                    "status_code": response.status_code,
                    "latency_ms": round(elapsed_ms, 2),
                    "endpoint_kind": getattr(request.state, "endpoint_kind", None),
                    "record_count": getattr(request.state, "record_count", None),
                },
            )
            return response
        finally:
            clear_request_context()
