from fastapi import APIRouter, Depends

from insurance_pricing.api.dependencies import get_prediction_service
from insurance_pricing.api.schemas import HealthResponse
from insurance_pricing.api.service import PredictionService

router = APIRouter(tags=["health"])


def _build_health_response(service: PredictionService) -> HealthResponse:
    return HealthResponse(**service.health())


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Confirms that the API process is running and that the configured model bundle was loaded "
        "during startup."
    ),
    operation_id="health_check",
    name="health_check",
    response_description="Operational status of the API process and the loaded model bundle.",
)
def health(service: PredictionService = Depends(get_prediction_service)) -> HealthResponse:
    return _build_health_response(service)


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description=(
        "Confirms that the API has completed startup and is ready to accept inference traffic from "
        "clients, reverse proxies, or orchestrators."
    ),
    operation_id="readiness_check",
    name="readiness_check",
    response_description="Readiness status of the API instance.",
)
def ready(service: PredictionService = Depends(get_prediction_service)) -> HealthResponse:
    return _build_health_response(service)
