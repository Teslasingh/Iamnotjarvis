"""Email list, detail, analysis, status, and archive routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

import storage
from api.deps import AnalyzeBatchBody, StatusBody, require_gmail_authorized
from errors import AppError
from services import analysis_service

router = APIRouter(prefix="/api", tags=["emails"])


@router.get("/emails")
def list_emails(
    status: str | None = None,
    include_non_jobs: bool = True,
    limit: int = 100,
    min_score: int | None = None,
    company: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    job_type: str | None = None,
) -> dict[str, Any]:
    return {
        "emails": storage.list_emails(
            status=status,
            include_non_jobs=include_non_jobs,
            limit=min(max(limit, 1), 250),
            min_score=min_score,
            company=company,
            date_from=date_from,
            date_to=date_to,
            job_type=job_type,
        )
    }


@router.post("/analyze/pending")
def analyze_pending(body: AnalyzeBatchBody) -> dict[str, Any]:
    return analysis_service.analyze_pending(limit=body.limit, force=body.force)


@router.get("/emails/{gmail_id}")
def get_email(gmail_id: str) -> dict[str, Any]:
    email = storage.get_email(gmail_id)
    if not email:
        raise HTTPException(404, "Email not found")
    email["draft"] = storage.latest_draft(gmail_id)
    return email


@router.post("/emails/{gmail_id}/analyze")
def analyze(gmail_id: str, force: bool = False) -> dict[str, Any]:
    try:
        return analysis_service.analyze_email_record(gmail_id, force=force)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


@router.post("/emails/{gmail_id}/status")
def update_status(gmail_id: str, body: StatusBody) -> dict[str, Any]:
    require_gmail_authorized()
    try:
        return analysis_service.update_email_status(gmail_id, body.status)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


@router.post("/emails/{gmail_id}/archive")
def archive(gmail_id: str) -> dict[str, Any]:
    require_gmail_authorized()
    try:
        return analysis_service.archive_email(gmail_id)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
