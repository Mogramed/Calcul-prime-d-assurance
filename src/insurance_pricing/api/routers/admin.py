from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from insurance_pricing.api.auth_store import (
    AdminUserSummaryRecord,
    StoredUserRecord,
    UserStore,
    UserStoreUnavailableError,
)
from insurance_pricing.api.dependencies import get_admin_user, get_quote_store, get_user_store
from insurance_pricing.api.quote_store import (
    AdminQuoteSummaryRecord,
    QuoteStore,
    QuoteStoreUnavailableError,
)
from insurance_pricing.api.schemas import (
    AdminQuoteListResponse,
    AdminQuoteSummaryResponse,
    AdminUserListResponse,
    AdminUserSummaryResponse,
)

router = APIRouter(tags=["admin"])


def _user_summary_response(record: AdminUserSummaryRecord) -> AdminUserSummaryResponse:
    return AdminUserSummaryResponse(
        id=record.id,
        created_at_utc=record.created_at_utc,
        email=record.email,
        role=record.role,
        is_active=record.is_active,
        email_verified_at_utc=record.email_verified_at_utc,
    )


def _quote_summary_response(record: AdminQuoteSummaryRecord) -> AdminQuoteSummaryResponse:
    return AdminQuoteSummaryResponse(
        id=record.id,
        created_at_utc=record.created_at_utc,
        run_id=record.run_id,
        type_contrat=record.type_contrat,
        marque_vehicule=record.marque_vehicule,
        modele_vehicule=record.modele_vehicule,
        prime_prediction=record.prime_prediction,
        user_id=record.user_id,
        owner_email=record.owner_email,
        deleted_at_utc=record.deleted_at_utc,
    )


@router.get(
    "/admin/users",
    response_model=AdminUserListResponse,
    summary="List user accounts for administrators",
    operation_id="admin_list_users",
)
async def admin_list_users(
    admin_user: StoredUserRecord = Depends(get_admin_user),
    user_store: UserStore = Depends(get_user_store),
) -> AdminUserListResponse:
    del admin_user
    try:
        users = await user_store.list_admin_users()
    except UserStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="User listing is unavailable.") from exc
    return AdminUserListResponse(
        count=len(users),
        users=[_user_summary_response(record) for record in users],
    )


@router.delete(
    "/admin/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate a user account",
    operation_id="admin_deactivate_user",
    responses={404: {"description": "User not found."}},
)
async def admin_deactivate_user(
    user_id: str,
    admin_user: StoredUserRecord = Depends(get_admin_user),
    user_store: UserStore = Depends(get_user_store),
) -> Response:
    if admin_user.id == user_id:
        raise HTTPException(status_code=400, detail="Administrators cannot deactivate their own account.")
    try:
        user = await user_store.deactivate_user(user_id)
    except UserStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="User update is unavailable.") from exc
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/admin/quotes",
    response_model=AdminQuoteListResponse,
    summary="List recent quotes for administrators",
    operation_id="admin_list_quotes",
)
async def admin_list_quotes(
    limit: int = Query(default=100, ge=1, le=500),
    admin_user: StoredUserRecord = Depends(get_admin_user),
    quote_store: QuoteStore = Depends(get_quote_store),
) -> AdminQuoteListResponse:
    del admin_user
    try:
        quotes = await quote_store.list_admin_quotes(limit=limit)
    except QuoteStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Quote listing is unavailable.") from exc
    return AdminQuoteListResponse(
        count=len(quotes),
        quotes=[_quote_summary_response(record) for record in quotes],
    )


@router.delete(
    "/admin/quotes/{quote_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a persisted quote",
    operation_id="admin_delete_quote",
    responses={404: {"description": "Quote not found."}},
)
async def admin_delete_quote(
    quote_id: str,
    admin_user: StoredUserRecord = Depends(get_admin_user),
    quote_store: QuoteStore = Depends(get_quote_store),
) -> Response:
    del admin_user
    try:
        quote = await quote_store.delete_quote(quote_id)
    except QuoteStoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Quote deletion is unavailable.") from exc
    if quote is None:
        raise HTTPException(status_code=404, detail="Quote not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
