"""Auth and status routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

import gmail_client
import storage
from config import CLIENT_SECRET_FILE, settings
from errors import AppError
from profile_store import load_persisted_profile, resume_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])


@router.get("/auth/start")
def auth_start() -> RedirectResponse:
    if not CLIENT_SECRET_FILE.exists():
        raise HTTPException(500, f"Missing Gmail client secret: {CLIENT_SECRET_FILE.name}")
    try:
        return RedirectResponse(gmail_client.authorization_url())
    except Exception as exc:
        logger.exception("Failed to start Gmail OAuth")
        raise HTTPException(500, f"Could not start Gmail authorization: {exc}") from exc


@router.get("/oauth2callback")
def oauth_callback(request: Request) -> RedirectResponse:
    try:
        gmail_client.fetch_token(str(request.url))
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    except Exception as exc:
        logger.exception("OAuth callback failed")
        raise HTTPException(400, f"Could not complete Gmail authorization: {exc}") from exc
    return RedirectResponse("/?auth=connected")


@router.get("/api/status")
def status() -> dict[str, Any]:
    profile = load_persisted_profile()
    current_resume = resume_path(profile)
    gmail_ready = False
    gmail_email = ""
    if gmail_client.is_authorized():
        gmail_ready = True
        try:
            gmail_email = gmail_client.gmail_profile().get("emailAddress", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gmail profile lookup failed: %s", exc)
            gmail_ready = False
    return {
        "gmail_authorized": gmail_ready,
        "gmail_email": gmail_email,
        "client_secret_exists": CLIENT_SECRET_FILE.exists(),
        "openai_configured": settings.llm_configured,
        "profile_ready": bool(profile.get("full_name") and profile.get("skills")),
        "resume_uploaded": current_resume is not None,
        "resume_file_name": current_resume.name if current_resume else "",
        "resume_profile_db_id": (profile.get("parsed_profile") or {}).get("db_id"),
        "profile_summary": profile.get("profile_summary", ""),
        "sync_info": storage.get_sync_info(),
    }
