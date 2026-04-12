from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from insurance_pricing.api.auth_store import StoredUserRecord
from insurance_pricing.api.dependencies import (
    get_client_id,
    get_current_user,
    get_prediction_service,
    get_quote_email_sender,
    get_quote_store,
)
from insurance_pricing.api.logging import get_logger
from insurance_pricing.api.quote_emailing import QuoteEmailDeliveryRecord, QuoteEmailSender
from insurance_pricing.api.quote_store import (
    QuoteCreateRecord,
    QuoteStore,
    QuoteStoreUnavailableError,
    QuoteSummaryRecord,
    StoredQuoteRecord,
    hash_client_id,
)
from insurance_pricing.api.reporting import build_quote_report_pdf
from insurance_pricing.api.schemas import (
    SINGLE_PREDICTION_EXAMPLE,
    PredictionInput,
    QuoteEmailDeliveryResponse,
    QuoteListResponse,
    QuoteResponse,
    QuoteResultResponse,
    QuoteSummaryResponse,
)
from insurance_pricing.api.service import PredictionService

router = APIRouter(tags=["quotes"])
QUOTE_ROUTER_LOGGER = get_logger("insurance_pricing.api.routers.quotes")


def _mark_request(request: Request, *, endpoint_kind: str) -> None:
    request.state.endpoint_kind = endpoint_kind
    request.state.record_count = 1


def _quote_result(prediction: dict[str, object]) -> QuoteResultResponse:
    return QuoteResultResponse(**prediction)


def _quote_response(
    record: StoredQuoteRecord,
    *,
    email_delivery: QuoteEmailDeliveryRecord | None = None,
) -> QuoteResponse:
    return QuoteResponse(
        id=record.id,
        created_at_utc=record.created_at_utc,
        run_id=record.run_id,
        input_payload=PredictionInput.model_validate(record.input_payload),
        result=_quote_result(
            {
                "frequency_prediction": record.frequency_prediction,
                "severity_prediction": record.severity_prediction,
                "prime_prediction": record.prime_prediction,
            }
        ),
        email_delivery=(
            QuoteEmailDeliveryResponse(
                status=email_delivery.status,
                recipient_email=email_delivery.recipient_email,
            )
            if email_delivery is not None
            else None
        ),
    )


def _quote_summary_response(record: QuoteSummaryRecord) -> QuoteSummaryResponse:
    return QuoteSummaryResponse(
        id=record.id,
        created_at_utc=record.created_at_utc,
        run_id=record.run_id,
        type_contrat=record.type_contrat,
        marque_vehicule=record.marque_vehicule,
        modele_vehicule=record.modele_vehicule,
        prime_prediction=record.prime_prediction,
    )


async def _get_accessible_quote(
    *,
    quote_store: QuoteStore,
    quote_id: str,
    client_id: str,
    current_user: StoredUserRecord | None,
) -> StoredQuoteRecord | None:
    if current_user is not None and current_user.role == "admin":
        return await quote_store.get_any_quote(quote_id)
    if current_user is not None:
        return await quote_store.get_user_quote(quote_id=quote_id, user_id=current_user.id)
    return await quote_store.get_quote(quote_id=quote_id, client_id_hash=hash_client_id(client_id))


@router.post(
    "/quotes",
    response_model=QuoteResponse,
    response_model_exclude_none=True,
    summary="Create a persisted quote",
    description=(
        "Scores one insurance record, persists the input payload and the resulting quote, and "
        "returns the stored quote detail for the current customer context."
    ),
    operation_id="create_quote",
    response_description="Persisted quote detail for the current customer context.",
    responses={
        400: {"description": "The required X-Client-ID header is missing or invalid."},
        503: {"description": "Quote persistence is unavailable because PostgreSQL is not ready."},
    },
)
async def create_quote(
    request: Request,
    payload: PredictionInput = Body(
        description=(
            "Raw business-facing fields for one insurance contract. The payload mirrors the "
            "technical prediction contract used by the model."
        ),
        openapi_examples={
            "single_record": {
                "summary": "Single contract quote example",
                "value": SINGLE_PREDICTION_EXAMPLE,
            }
        },
    ),
    client_id: str = Depends(get_client_id),
    current_user: StoredUserRecord | None = Depends(get_current_user),
    service: PredictionService = Depends(get_prediction_service),
    quote_store: QuoteStore = Depends(get_quote_store),
    quote_email_sender: QuoteEmailSender = Depends(get_quote_email_sender),
) -> QuoteResponse:
    _mark_request(request, endpoint_kind="quote_create")
    payload_obj = payload.model_dump(mode="python")
    prediction = await run_in_threadpool(service.predict_record, payload_obj)

    try:
        stored_quote = await quote_store.create_quote(
            QuoteCreateRecord(
                client_id_hash=hash_client_id(client_id),
                user_id=current_user.id if current_user is not None else None,
                run_id=service.run_id,
                input_payload=payload_obj,
                frequency_prediction=float(prediction["frequency_prediction"]),
                severity_prediction=float(prediction["severity_prediction"]),
                prime_prediction=float(prediction["prime_prediction"]),
            )
        )
    except QuoteStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Quote persistence is unavailable.") from exc

    email_delivery = await _send_quote_email_if_possible(
        stored_quote=stored_quote,
        current_user=current_user,
        quote_email_sender=quote_email_sender,
    )

    return _quote_response(stored_quote, email_delivery=email_delivery)


@router.get(
    "/quotes",
    response_model=QuoteListResponse,
    response_model_exclude_none=True,
    summary="List persisted quotes",
    description=(
        "Returns the quote history associated with the current browser context or authenticated "
        "account, ordered from the most recent to the oldest quote."
    ),
    operation_id="list_quotes",
    name="list_quotes",
    response_description="Persisted quote history for the current customer context.",
    responses={
        400: {"description": "The required X-Client-ID header is missing or invalid."},
        503: {"description": "Quote persistence is unavailable because PostgreSQL is not ready."},
    },
)
async def list_quotes(
    request: Request,
    client_id: str = Depends(get_client_id),
    current_user: StoredUserRecord | None = Depends(get_current_user),
    quote_store: QuoteStore = Depends(get_quote_store),
) -> QuoteListResponse:
    _mark_request(request, endpoint_kind="quote_list")
    try:
        if current_user is not None:
            quote_records = await quote_store.list_user_quotes(current_user.id)
        else:
            quote_records = await quote_store.list_quotes(hash_client_id(client_id))
    except QuoteStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Quote persistence is unavailable.") from exc

    return QuoteListResponse(
        count=len(quote_records),
        quotes=[_quote_summary_response(record) for record in quote_records],
    )


@router.get(
    "/quotes/{quote_id}",
    response_model=QuoteResponse,
    response_model_exclude_none=True,
    summary="Get a persisted quote",
    description=(
        "Returns one persisted quote when it belongs to the current browser context, the current "
        "account, or when it is accessed by an administrator."
    ),
    operation_id="get_quote",
    response_description="Persisted quote detail for the current customer context.",
    responses={
        400: {"description": "The required X-Client-ID header is missing or invalid."},
        404: {"description": "The quote does not exist for the current customer context."},
        503: {"description": "Quote persistence is unavailable because PostgreSQL is not ready."},
    },
)
async def get_quote(
    quote_id: str,
    request: Request,
    client_id: str = Depends(get_client_id),
    current_user: StoredUserRecord | None = Depends(get_current_user),
    quote_store: QuoteStore = Depends(get_quote_store),
) -> QuoteResponse:
    _mark_request(request, endpoint_kind="quote_get")
    try:
        stored_quote = await _get_accessible_quote(
            quote_store=quote_store,
            quote_id=quote_id,
            client_id=client_id,
            current_user=current_user,
        )
    except QuoteStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Quote persistence is unavailable.") from exc

    if stored_quote is None or stored_quote.deleted_at_utc is not None:
        raise HTTPException(status_code=404, detail="Quote not found.")
    return _quote_response(stored_quote)


@router.get(
    "/quotes/{quote_id}/report.pdf",
    summary="Generate the PDF report for one quote",
    operation_id="download_quote_report",
    responses={
        200: {"content": {"application/pdf": {}}},
        404: {"description": "Quote not found."},
        503: {"description": "Quote report generation is unavailable."},
    },
)
async def download_quote_report(
    quote_id: str,
    request: Request,
    client_id: str = Depends(get_client_id),
    current_user: StoredUserRecord | None = Depends(get_current_user),
    quote_store: QuoteStore = Depends(get_quote_store),
) -> Response:
    _mark_request(request, endpoint_kind="quote_report_download")
    try:
        quote = await _get_accessible_quote(
            quote_store=quote_store,
            quote_id=quote_id,
            client_id=client_id,
            current_user=current_user,
        )
    except QuoteStoreUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail="Quote report generation is unavailable."
        ) from exc

    if quote is None or quote.deleted_at_utc is not None:
        raise HTTPException(status_code=404, detail="Quote not found.")

    pdf_bytes = await run_in_threadpool(build_quote_report_pdf, quote)
    filename = _pdf_filename(quote)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _pdf_filename(quote: StoredQuoteRecord) -> str:
    return f"{Path('nova-devis-' + quote.id).stem}.pdf"


async def _send_quote_email_if_possible(
    *,
    stored_quote: StoredQuoteRecord,
    current_user: StoredUserRecord | None,
    quote_email_sender: QuoteEmailSender,
) -> QuoteEmailDeliveryRecord:
    if current_user is None:
        return QuoteEmailDeliveryRecord(status="skipped", recipient_email=None)

    try:
        pdf_bytes = await run_in_threadpool(build_quote_report_pdf, stored_quote)
        return await quote_email_sender.send_quote_email(
            quote=stored_quote,
            recipient_email=current_user.email,
            pdf_bytes=pdf_bytes,
        )
    except Exception:
        QUOTE_ROUTER_LOGGER.warning(
            "quote_email_delivery_failed",
            extra={
                "quote_id": stored_quote.id,
                "recipient_email": current_user.email,
            },
            exc_info=True,
        )
        return QuoteEmailDeliveryRecord(
            status="failed",
            recipient_email=current_user.email,
        )
