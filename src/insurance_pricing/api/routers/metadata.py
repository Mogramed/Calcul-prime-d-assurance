from fastapi import APIRouter, Depends, Request

from insurance_pricing import __version__
from insurance_pricing.api.dependencies import get_prediction_service
from insurance_pricing.api.schemas import ApiIndexResponse, ModelMetadataResponse, VersionResponse
from insurance_pricing.api.service import PredictionService

router = APIRouter(tags=["metadata"])


@router.get(
    "/",
    response_model=ApiIndexResponse,
    response_model_exclude_none=True,
    summary="API entrypoint",
    description=(
        "Quick navigation endpoint exposing the main operational and documentation URLs of the "
        "current API instance."
    ),
    operation_id="api_index",
    response_description="Top-level navigation links for the current API instance.",
)
def api_index(request: Request) -> ApiIndexResponse:
    return ApiIndexResponse(
        name="Insurance Pricing API",
        api_version=__version__,
        docs_url=str(request.url_for("swagger_ui_html")),
        redoc_url=str(request.url_for("redoc_html")),
        openapi_url=str(request.url_for("openapi")),
        health_url=str(request.url_for("health_check")),
        ready_url=str(request.url_for("readiness_check")),
        version_url=str(request.url_for("get_version")),
        current_model_url=str(request.url_for("get_current_model")),
        prediction_schema_url=str(request.url_for("get_prediction_schema")),
        quotes_url=str(request.url_for("list_quotes")),
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    response_model_exclude_none=True,
    summary="API and model version",
    description=(
        "Returns the installed package version together with the model run identifier configured at "
        "application startup."
    ),
    operation_id="get_version",
    name="get_version",
    response_description="Package version and currently served model run identifier.",
)
def version(service: PredictionService = Depends(get_prediction_service)) -> VersionResponse:
    return VersionResponse(api_version=__version__, model_run_id=service.run_id)


@router.get(
    "/models/current",
    response_model=ModelMetadataResponse,
    response_model_exclude_none=True,
    summary="Current served model metadata",
    description=(
        "Returns the metadata of the model bundle currently loaded in memory, including artifacts, "
        "metrics, feature schema, and training configuration summary."
    ),
    operation_id="get_current_model",
    response_description="Metadata describing the model bundle currently served by this API instance.",
)
def get_current_model(
    service: PredictionService = Depends(get_prediction_service),
) -> ModelMetadataResponse:
    return ModelMetadataResponse(**service.current_model_metadata())
