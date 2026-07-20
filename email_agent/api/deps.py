"""Shared API dependencies and schemas."""

from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel, Field

from errors import AppError
from services.sync_service import require_gmail_authorized as _require_gmail


class SyncBody(BaseModel):
    analyze_new: bool = True


class AnalyzeBatchBody(BaseModel):
    limit: int = Field(default=25, ge=1, le=100)
    force: bool = False


class DraftUpdateBody(BaseModel):
    subject: str
    body: str
    attach_resume: bool = False


class StatusBody(BaseModel):
    status: str


class AgentPromptBody(BaseModel):
    agent_prompt: str


def require_gmail_authorized() -> None:
    try:
        _require_gmail()
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


def raise_http(exc: Exception) -> None:
    """Re-raise domain errors as HTTPException; leave others untouched."""
    if isinstance(exc, AppError):
        raise HTTPException(exc.status_code, exc.message) from exc
    raise exc
