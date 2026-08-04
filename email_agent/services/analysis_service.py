"""Email analysis orchestration."""

from __future__ import annotations

import logging
from typing import Any

import ai_agent
import gmail_client
import prompt_store
import storage
from constants import ANALYSIS_STATUSES
from errors import ExternalServiceError, NotFoundError
from profile_store import load_persisted_profile

logger = logging.getLogger(__name__)


def analyze_email_record(gmail_id: str, *, force: bool = False) -> dict[str, Any]:
    email = storage.get_email(gmail_id)
    if not email:
        raise NotFoundError("Email not found")

    if not storage.should_analyze(gmail_id, force=force):
        email["skipped_duplicate_analysis"] = True
        return email

    storage.set_analysis_status(gmail_id, "Analyzing")
    try:
        agent_prompt = prompt_store.load_agent_prompt()
        analysis = ai_agent.analyze_email(email, load_persisted_profile(), agent_prompt=agent_prompt)
        storage.save_analysis(gmail_id, analysis)
        saved = storage.get_email(gmail_id) or {}
        _maybe_apply_label(gmail_id, saved.get("status"))
        logger.info(
            "Analyzed email %s → status=%s score=%s",
            gmail_id,
            saved.get("status"),
            saved.get("match_score"),
        )
        return saved
    except Exception as exc:
        storage.set_analysis_status(gmail_id, "Not Analyzed")
        logger.exception("Analysis failed for %s", gmail_id)
        raise ExternalServiceError(f"Analysis failed: {exc}", status_code=500) from exc


def analyze_pending(*, limit: int = 25, force: bool = False) -> dict[str, Any]:
    ids = storage.pending_email_ids(limit=min(max(limit, 1), 100))
    analyzed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for gmail_id in ids:
        try:
            analyzed.append(analyze_email_record(gmail_id, force=force))
        except Exception as exc:  # noqa: BLE001
            message = getattr(exc, "message", str(exc))
            logger.warning("Pending analysis failed for %s: %s", gmail_id, message)
            errors.append({"gmail_id": gmail_id, "message": str(message)})

    return {
        "requested": len(ids),
        "analyzed": len(analyzed),
        "emails": analyzed,
        "errors": errors[:10],
        "metrics": storage.dashboard_metrics(),
    }


def analyze_all_pending(
    *,
    track_progress: bool = False,
    total_pending: int = 0,
    on_progress: Any | None = None,
) -> tuple[int, list[dict[str, str]]]:
    analyzed = 0
    errors: list[dict[str, str]] = []
    pending_total = total_pending or storage.count_pending_emails()

    while True:
        pending_ids = storage.pending_email_ids(limit=50)
        if not pending_ids:
            break
        for gmail_id in pending_ids:
            try:
                analyze_email_record(gmail_id, force=False)
                analyzed += 1
                if track_progress and pending_total and callable(on_progress):
                    percent = 55 + int((analyzed / pending_total) * 45)
                    on_progress(
                        percent,
                        "analyze",
                        f"Analyzing emails ({analyzed}/{pending_total})...",
                        analyzed=analyzed,
                        pending_total=pending_total,
                    )
            except Exception as exc:  # noqa: BLE001
                message = getattr(exc, "message", str(exc))
                errors.append({"gmail_id": gmail_id, "message": str(message)})

    return analyzed, errors


def update_email_status(gmail_id: str, status: str) -> dict[str, Any]:
    if status not in ANALYSIS_STATUSES:
        from errors import ValidationError

        raise ValidationError("Invalid status")
    storage.update_email_status(gmail_id, status)
    _maybe_apply_label(gmail_id, status)
    return storage.get_email(gmail_id) or {}


def archive_email(gmail_id: str) -> dict[str, Any]:
    gmail_client.archive_message(gmail_id)
    storage.mark_archived(gmail_id)
    logger.info("Archived email %s", gmail_id)
    return {"archived": True}


def apply_gmail_status_label(gmail_id: str, status: str | None) -> None:
    """Best-effort Gmail label update; never raises to callers."""
    _maybe_apply_label(gmail_id, status)


def _maybe_apply_label(gmail_id: str, status: str | None) -> None:
    if not status or status not in gmail_client.JOB_LABELS:
        return
    if not gmail_client.is_authorized():
        return
    try:
        gmail_client.apply_status_label(gmail_id, status)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not apply Gmail label for %s (%s): %s", gmail_id, status, exc)
