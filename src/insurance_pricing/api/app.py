from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

# --- Nos nouveaux imports pour la base de données ---
from .database import engine
from . import db_models
# ----------------------------------------------------

from insurance_pricing import __version__
from insurance_pricing.api.dependencies import get_settings
from insurance_pricing.api.middleware import install_observability_middleware
from insurance_pricing.api.routers.health import router as health_router
from insurance_pricing.api.routers.metadata import router as metadata_router
from insurance_pricing.api.routers.predict import router as predict_router
from insurance_pricing.api.service import PredictionService
from insurance_pricing.api.settings import AppSettings


def create_app(settings: AppSettings | None = None) -> FastAPI:
    resolved_settings = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # --- CRÉATION DE LA TABLE POSTGRESQL AU DÉMARRAGE ---
        db_models.Base.metadata.create_all(bind=engine)
        # ----------------------------------------------------
        
        try:
            app.state.prediction_service = PredictionService.load(resolved_settings.run_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load prediction service for run_id '{resolved_settings.run_id}'."
            ) from exc
        yield

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
            "provided by the caller."
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
        ],
    )
    app.state.settings = resolved_settings
    install_observability_middleware(app)
    app.include_router(metadata_router)
    app.include_router(health_router)
    app.include_router(predict_router)
    return app