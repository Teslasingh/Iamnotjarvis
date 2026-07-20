"""Draft routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from api.deps import DraftUpdateBody
from errors import AppError
from services import draft_service

router = APIRouter(prefix="/api", tags=["drafts"])


@router.post("/emails/{gmail_id}/draft")
def draft(gmail_id: str) -> dict[str, Any]:
    try:
        return draft_service.create_draft(gmail_id)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


@router.put("/drafts/{draft_id}")
def update_draft(draft_id: int, body: DraftUpdateBody) -> dict[str, Any]:
    try:
        return draft_service.update_draft(
            draft_id,
            subject=body.subject,
            body=body.body,
            attach_resume=body.attach_resume,
        )
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


@router.post("/drafts/{draft_id}/send")
def send_draft(draft_id: int) -> dict[str, Any]:
    try:
        return draft_service.send_draft(draft_id)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
