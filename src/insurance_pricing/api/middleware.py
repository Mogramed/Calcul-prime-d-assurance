from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"


def install_observability_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def add_request_observability(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error.",
                    "request_id": request_id,
                },
            )
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.2f}"
            return response
        else:
            elapsed_ms = (perf_counter() - started_at) * 1000.0
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.2f}"
            return response
