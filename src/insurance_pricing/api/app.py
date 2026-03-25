from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from insurance_pricing import __version__
from insurance_pricing.api.audit import AuditStore
from insurance_pricing.api.auth_store import UserStore
from insurance_pricing.api.db import PostgresAuditStore
from insurance_pricing.api.dependencies import get_settings
from insurance_pricing.api.errors import install_exception_handlers
from insurance_pricing.api.logging import configure_logging, get_logger
from insurance_pricing.api.middleware import install_observability_middleware
from insurance_pricing.api.quote_store import QuoteStore
from insurance_pricing.api.routers.admin import router as admin_router
from insurance_pricing.api.routers.auth import router as auth_router
from insurance_pricing.api.routers.health import router as health_router
from insurance_pricing.api.routers.metadata import router as metadata_router
from insurance_pricing.api.routers.predict import router as predict_router
from insurance_pricing.api.routers.quotes import router as quotes_router
from insurance_pricing.api.service import PredictionService
from insurance_pricing.api.settings import AppSettings

APP_LOGGER = get_logger("insurance_pricing.api.app")


def create_app(
    settings: AppSettings | None = None,
    *,
    prediction_service: PredictionService | None = None,
    audit_store: AuditStore | None = None,
    quote_store: QuoteStore | None = None,
    user_store: UserStore | None = None,
) -> FastAPI:
    resolved_settings = settings if settings is not None else get_settings()
    configure_logging(level=resolved_settings.log_level, json_logs=resolved_settings.log_json)
    resolved_audit_store = (
        audit_store
        if audit_store is not None
        else PostgresAuditStore(resolved_settings.database_url)
    )
    resolved_quote_store = quote_store if quote_store is not None else resolved_audit_store
    resolved_user_store = user_store if user_store is not None else resolved_audit_store

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.prediction_service = (
                prediction_service
                if prediction_service is not None
                else PredictionService.load(resolved_settings.run_id)
            )
            app.state.audit_store = resolved_audit_store
            app.state.quote_store = resolved_quote_store
            app.state.user_store = resolved_user_store
            await resolved_audit_store.startup()
            if resolved_quote_store is not resolved_audit_store:
                await resolved_quote_store.startup()
            if resolved_user_store not in {resolved_audit_store, resolved_quote_store}:
                await resolved_user_store.startup()
        except Exception as exc:
            if resolved_user_store not in {resolved_audit_store, resolved_quote_store}:
                await resolved_user_store.shutdown()
            if resolved_quote_store is not resolved_audit_store:
                await resolved_quote_store.shutdown()
            await resolved_audit_store.shutdown()
            raise RuntimeError(
                f"Failed to initialize the API runtime for run_id '{resolved_settings.run_id}'."
            ) from exc
        APP_LOGGER.info(
            "api_startup_completed",
            extra={
                "run_id": resolved_settings.run_id,
                "database_configured": True,
                "cors_allowed_origins": resolved_settings.cors_allowed_origins,
                "admin_emails": resolved_settings.admin_emails,
            },
        )
        yield
        if resolved_user_store not in {resolved_audit_store, resolved_quote_store}:
            await resolved_user_store.shutdown()
        if resolved_quote_store is not resolved_audit_store:
            await resolved_quote_store.shutdown()
        await resolved_audit_store.shutdown()
        APP_LOGGER.info("api_shutdown_completed", extra={"run_id": resolved_settings.run_id})

    app = FastAPI(
        title="Insurance Pricing API",
        summary="HTTP inference API for the insurance premium pricing model.",
        description=(
            "Production-style FastAPI surface for serving insurance pricing predictions.\n\n"
            "## Scope\n"
            "- score raw insurance business records\n"
            "- expose frequency, severity, and final premium predictions\n"
            "- keep the served model bundle pinned at startup through `INSURANCE_PRICING_RUN_ID`\n\n"
            "## Request Contract\n"
            "Send the raw columns from the historical scoring dataset, such as the ones present in "
            "`data/test.csv`. Feature engineering stays inside the service layer, so clients only "
            "need to send business-facing fields.\n\n"
            "## Response Headers\n"
            "Every response includes `X-Request-ID` and `X-Process-Time-Ms` headers to help trace "
            "calls across clients, logs, and deployment environments.\n\n"
            "## Batch Behavior\n"
            "Batch endpoints preserve input order and return the optional `index` field when it is "
            "provided by the caller.\n\n"
            "## Product Quote Endpoints\n"
            "The public `/quotes` endpoints persist end-user quote history scoped to an anonymous "
            "`X-Client-ID` header so the Next.js frontend can retrieve prior simulations."
        ),
        version=__version__,
        lifespan=lifespan,
        contact={
            "name": "Insurance Pricing Team",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "docExpansion": "list",
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "defaultModelRendering": "model",
            "filter": True,
            "tryItOutEnabled": True,
        },
        openapi_tags=[
            {
                "name": "metadata",
                "description": (
                    "Discovery and model metadata endpoints used to identify the running API version "
                    "and the currently loaded model bundle."
                ),
            },
            {
                "name": "health",
                "description": "Operational health and readiness endpoints for deployment probes.",
            },
            {
                "name": "predict",
                "description": (
                    "Inference endpoints for scoring one or many insurance records with either "
                    "component-level or full-premium outputs."
                ),
            },
            {
                "name": "quotes",
                "description": (
                    "Public quote workflow endpoints used by the product frontend to create and "
                    "retrieve end-user quote history."
                ),
            },
            {
                "name": "auth",
                "description": "Account registration, login, logout, and session discovery.",
            },
            {
                "name": "admin",
                "description": "Protected administrative endpoints for accounts and quote moderation.",
            },
        ],
    )
    app.state.settings = resolved_settings
    app.state.quote_store = resolved_quote_store
    app.state.user_store = resolved_user_store
    if resolved_settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_allowed_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    install_observability_middleware(app)
    install_exception_handlers(app)
    app.include_router(metadata_router)
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(auth_router)
    app.include_router(quotes_router)
    app.include_router(admin_router)
    return app
