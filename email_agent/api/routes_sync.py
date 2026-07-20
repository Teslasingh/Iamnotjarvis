"""Sync and metrics routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

import storage
from api.deps import SyncBody
from errors import AppError
from services import sync_service

router = APIRouter(prefix="/api", tags=["sync"])


@router.post("/sync")
def sync_messages(body: SyncBody | None = None) -> dict[str, Any]:
    analyze_new = True if body is None else body.analyze_new
    try:
        return sync_service.start_background_sync(analyze_new=analyze_new)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


@router.get("/sync/progress")
def sync_progress() -> dict[str, Any]:
    return sync_service.progress_snapshot()


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    return storage.dashboard_metrics()


@router.get("/filter-options")
def filter_options() -> dict[str, Any]:
    return storage.filter_options()
