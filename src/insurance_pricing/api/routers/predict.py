from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.concurrency import run_in_threadpool

from insurance_pricing.api.audit import (
    AuditStore,
    PredictionAuditRecord,
    PredictionOutputRecord,
    hash_payload,
)
from insurance_pricing.api.dependencies import get_audit_store, get_prediction_service
from insurance_pricing.api.schemas import (
    BATCH_PREDICTION_EXAMPLE,
    SINGLE_PREDICTION_EXAMPLE,
    FrequencyPredictionBatchResponse,
    FrequencyPredictionResponse,
    PredictionBatchInput,
    PredictionFieldDescriptor,
    PredictionInput,
    PredictionSchemaResponse,
    PrimePredictionBatchResponse,
    PrimePredictionResponse,
    SeverityPredictionBatchResponse,
    SeverityPredictionResponse,
)
from insurance_pricing.api.service import PredictionService

router = APIRouter(tags=["predict"])

PREDICTION_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {
        "description": (
            "Payload validation failed. The request must match the raw business fields expected by "
            "the model input contract."
        )
    },
    500: {
        "description": (
            "The model bundle could not score the request. Check the payload semantics and the "
            "loaded artifact integrity."
        )
    },
    503: {
        "description": "Prediction persistence is unavailable because PostgreSQL is not ready.",
    },
}


def _schema_type(schema: dict[str, Any]) -> str:
    if "type" in schema:
        schema_type = schema["type"]
        if isinstance(schema_type, list):
            return " | ".join(str(item) for item in schema_type)
        return str(schema_type)
    if "anyOf" in schema:
        return " | ".join(_schema_type(item) for item in schema["anyOf"])
    return "object"


def _mark_request(request: Request, *, endpoint_kind: str, record_count: int) -> None:
    request.state.endpoint_kind = endpoint_kind
    request.state.record_count = record_count


def _latency_ms(request: Request) -> float:
    started_at = getattr(request.state, "started_at", perf_counter())
    return round((perf_counter() - started_at) * 1000.0, 2)


def _build_prediction_outputs(
    predictions: Sequence[Mapping[str, Any]],
) -> list[PredictionOutputRecord]:
    return [
        PredictionOutputRecord(
            record_position=position,
            frequency_prediction=_as_float(prediction.get("frequency_prediction")),
            severity_prediction=_as_float(prediction.get("severity_prediction")),
            prime_prediction=_as_float(prediction.get("prime_prediction")),
        )
        for position, prediction in enumerate(predictions)
    ]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


async def _persist_prediction_audit(
    *,
    request: Request,
    audit_store: AuditStore,
    service: PredictionService,
    payload: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    status_code: int = 200,
) -> None:
    await audit_store.persist_prediction(
        PredictionAuditRecord(
            request_id=request.state.request_id,
            endpoint=request.url.path,
            run_id=service.run_id,
            record_count=len(predictions),
            payload_hash=hash_payload(payload),
            status_code=status_code,
            latency_ms=_latency_ms(request),
            outputs=_build_prediction_outputs(predictions),
        )
    )


@router.get(
    "/predict/schema",
    response_model=PredictionSchemaResponse,
    response_model_exclude_none=True,
    summary="Prediction input contract",
    description=(
        "Returns a client-friendly summary of the raw fields expected by the prediction endpoints, "
        "including required vs optional fields. Use this endpoint before integrating a client, a "
        "frontend form, or a batch scoring job."
    ),
    operation_id="get_prediction_schema",
    name="get_prediction_schema",
    response_description="Summary of the single-record and batch input contract for prediction endpoints.",
)
def get_prediction_schema() -> PredictionSchemaResponse:
    record_schema = PredictionInput.model_json_schema()
    field_properties = record_schema.get("properties", {})
    required_fields = list(record_schema.get("required", []))
    optional_fields = [name for name in field_properties if name not in required_fields]

    return PredictionSchemaResponse(
        record_model=record_schema.get("title", "PredictionInput"),
        batch_model=PredictionBatchInput.model_json_schema().get("title", "PredictionBatchInput"),
        supports_batch=True,
        required_fields=required_fields,
        optional_fields=optional_fields,
        fields=[
            PredictionFieldDescriptor(
                name=name,
                type=_schema_type(field_schema),
                required=name in required_fields,
                description=field_schema.get("description"),
            )
            for name, field_schema in field_properties.items()
        ],
    )


@router.post(
    "/predict/frequency",
    response_model=FrequencyPredictionResponse,
    response_model_exclude_none=True,
    summary="Predict claim frequency",
    description=(
        "Scores one insurance record and returns only the calibrated frequency component.\n\n"
        "This endpoint is useful when you want to inspect the claim occurrence probability without "
        "combining it with the severity model."
    ),
    operation_id="predict_frequency",
    response_description="Frequency prediction for a single insurance record.",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict_frequency(
    request: Request,
    payload: PredictionInput = Body(
        description="Raw business-facing fields for one insurance contract.",
        openapi_examples={
            "single_record": {
                "summary": "Single contract scoring example",
                "value": SINGLE_PREDICTION_EXAMPLE,
            }
        },
    ),
    service: PredictionService = Depends(get_prediction_service),
    audit_store: AuditStore = Depends(get_audit_store),
) -> FrequencyPredictionResponse:
    _mark_request(request, endpoint_kind="frequency", record_count=1)
    payload_obj = payload.model_dump(mode="python")
    prediction = await run_in_threadpool(service.predict_frequency_record, payload_obj)
    await _persist_prediction_audit(
        request=request,
        audit_store=audit_store,
        service=service,
        payload=payload_obj,
        predictions=[prediction],
    )
    return FrequencyPredictionResponse(**prediction)


@router.post(
    "/predict/frequency/batch",
    response_model=FrequencyPredictionBatchResponse,
    response_model_exclude_none=True,
    summary="Predict claim frequency in batch",
    description=(
        "Scores multiple insurance records and returns frequency predictions in the same order as "
        "the request payload."
    ),
    operation_id="predict_frequency_batch",
    response_description="Frequency predictions for multiple insurance records.",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict_frequency_batch(
    request: Request,
    payload: PredictionBatchInput = Body(
        description=(
            "Batch scoring payload for frequency inference. Input order is preserved in the "
            "response, which makes the endpoint suitable for bulk scoring pipelines."
        ),
        openapi_examples={
            "batch_records": {
                "summary": "Batch contract scoring example",
                "value": BATCH_PREDICTION_EXAMPLE,
            }
        },
    ),
    service: PredictionService = Depends(get_prediction_service),
    audit_store: AuditStore = Depends(get_audit_store),
) -> FrequencyPredictionBatchResponse:
    _mark_request(request, endpoint_kind="frequency", record_count=len(payload.records))
    payload_obj = payload.model_dump(mode="python")
    record_payloads = [record.model_dump(mode="python") for record in payload.records]
    predictions = await run_in_threadpool(service.predict_frequency_records, record_payloads)
    await _persist_prediction_audit(
        request=request,
        audit_store=audit_store,
        service=service,
        payload=payload_obj,
        predictions=predictions,
    )
    return FrequencyPredictionBatchResponse(
        run_id=service.run_id,
        count=len(predictions),
        predictions=[FrequencyPredictionResponse(**prediction) for prediction in predictions],
    )


@router.post(
    "/predict/severity",
    response_model=SeverityPredictionResponse,
    response_model_exclude_none=True,
    summary="Predict claim severity",
    description=(
        "Scores one insurance record and returns only the severity component.\n\n"
        "Use this endpoint when you want to inspect the conditional expected claim cost without the "
        "frequency layer."
    ),
    operation_id="predict_severity",
    response_description="Severity prediction for a single insurance record.",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict_severity(
    request: Request,
    payload: PredictionInput = Body(
        description=(
            "Raw business-facing fields for one insurance contract. The payload structure is the "
            "same as for the other prediction endpoints."
        ),
        openapi_examples={
            "single_record": {
                "summary": "Single contract scoring example",
                "value": SINGLE_PREDICTION_EXAMPLE,
            }
        },
    ),
    service: PredictionService = Depends(get_prediction_service),
    audit_store: AuditStore = Depends(get_audit_store),
) -> SeverityPredictionResponse:
    _mark_request(request, endpoint_kind="severity", record_count=1)
    payload_obj = payload.model_dump(mode="python")
    prediction = await run_in_threadpool(service.predict_severity_record, payload_obj)
    await _persist_prediction_audit(
        request=request,
        audit_store=audit_store,
        service=service,
        payload=payload_obj,
        predictions=[prediction],
    )
    return SeverityPredictionResponse(**prediction)


@router.post(
    "/predict/severity/batch",
    response_model=SeverityPredictionBatchResponse,
    response_model_exclude_none=True,
    summary="Predict claim severity in batch",
    description=(
        "Scores multiple insurance records and returns severity predictions in the same order as "
        "the request payload."
    ),
    operation_id="predict_severity_batch",
    response_description="Severity predictions for multiple insurance records.",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict_severity_batch(
    request: Request,
    payload: PredictionBatchInput = Body(
        description=(
            "Batch scoring payload for severity inference. Each input record produces exactly one "
            "output prediction in the returned `predictions` list."
        ),
        openapi_examples={
            "batch_records": {
                "summary": "Batch contract scoring example",
                "value": BATCH_PREDICTION_EXAMPLE,
            }
        },
    ),
    service: PredictionService = Depends(get_prediction_service),
    audit_store: AuditStore = Depends(get_audit_store),
) -> SeverityPredictionBatchResponse:
    _mark_request(request, endpoint_kind="severity", record_count=len(payload.records))
    payload_obj = payload.model_dump(mode="python")
    record_payloads = [record.model_dump(mode="python") for record in payload.records]
    predictions = await run_in_threadpool(service.predict_severity_records, record_payloads)
    await _persist_prediction_audit(
        request=request,
        audit_store=audit_store,
        service=service,
        payload=payload_obj,
        predictions=predictions,
    )
    return SeverityPredictionBatchResponse(
        run_id=service.run_id,
        count=len(predictions),
        predictions=[SeverityPredictionResponse(**prediction) for prediction in predictions],
    )


@router.post(
    "/predict/prime",
    response_model=PrimePredictionResponse,
    response_model_exclude_none=True,
    summary="Predict final premium",
    description=(
        "Scores one insurance record and returns the full pricing decomposition: frequency, "
        "severity, and final premium.\n\n"
        "This is the main endpoint to call when you want the final commercial premium output."
    ),
    operation_id="predict_prime",
    response_description="Frequency, severity, and premium predictions for a single insurance record.",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict_prime(
    request: Request,
    payload: PredictionInput = Body(
        description=(
            "Raw business-facing fields for one insurance contract. The server computes engineered "
            "features internally before running the model bundle."
        ),
        openapi_examples={
            "single_record": {
                "summary": "Single contract scoring example",
                "value": SINGLE_PREDICTION_EXAMPLE,
            }
        },
    ),
    service: PredictionService = Depends(get_prediction_service),
    audit_store: AuditStore = Depends(get_audit_store),
) -> PrimePredictionResponse:
    _mark_request(request, endpoint_kind="prime", record_count=1)
    payload_obj = payload.model_dump(mode="python")
    prediction = await run_in_threadpool(service.predict_record, payload_obj)
    await _persist_prediction_audit(
        request=request,
        audit_store=audit_store,
        service=service,
        payload=payload_obj,
        predictions=[prediction],
    )
    return PrimePredictionResponse(**prediction)


@router.post(
    "/predict/prime/batch",
    response_model=PrimePredictionBatchResponse,
    response_model_exclude_none=True,
    summary="Predict final premium in batch",
    description=(
        "Scores multiple insurance records and returns the full pricing decomposition for each one "
        "in the same order as the input payload."
    ),
    operation_id="predict_prime_batch",
    response_description="Frequency, severity, and premium predictions for multiple insurance records.",
    responses=PREDICTION_ERROR_RESPONSES,
)
async def predict_prime_batch(
    request: Request,
    payload: PredictionBatchInput = Body(
        description=(
            "Batch scoring payload for full premium inference. This endpoint is designed for bulk "
            "scoring while preserving the original order of the submitted contracts."
        ),
        openapi_examples={
            "batch_records": {
                "summary": "Batch contract scoring example",
                "value": BATCH_PREDICTION_EXAMPLE,
            }
        },
    ),
    service: PredictionService = Depends(get_prediction_service),
    audit_store: AuditStore = Depends(get_audit_store),
) -> PrimePredictionBatchResponse:
    _mark_request(request, endpoint_kind="prime", record_count=len(payload.records))
    payload_obj = payload.model_dump(mode="python")
    record_payloads = [record.model_dump(mode="python") for record in payload.records]
    predictions = await run_in_threadpool(service.predict_records, record_payloads)
    await _persist_prediction_audit(
        request=request,
        audit_store=audit_store,
        service=service,
        payload=payload_obj,
        predictions=predictions,
    )
    return PrimePredictionBatchResponse(
        run_id=service.run_id,
        count=len(predictions),
        predictions=[PrimePredictionResponse(**prediction) for prediction in predictions],
    )
