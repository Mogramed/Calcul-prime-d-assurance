from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

REQUEST_LOG_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_log_context",
    default=None,
)

_STANDARD_LOG_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(REQUEST_LOG_CONTEXT.get() or {})
        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_KEYS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(*, level: str, json_logs: bool) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger.addHandler(handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True


def bind_request_context(**context: Any) -> None:
    current = dict(REQUEST_LOG_CONTEXT.get() or {})
    current.update({key: value for key, value in context.items() if value is not None})
    REQUEST_LOG_CONTEXT.set(current)


def clear_request_context() -> None:
    REQUEST_LOG_CONTEXT.set(None)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
