"""Inbox sync orchestration with progress tracking."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable

import gmail_client
import storage
from constants import SYNC_MAX_EMAILS, SYNC_PAGE_SIZE, SYNC_WINDOW_DAYS
from errors import AuthorizationError, ExternalServiceError
from services.analysis_service import analyze_all_pending

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str, str], None]

_sync_lock = threading.Lock()
_sync_state: dict[str, Any] = {
    "active": False,
    "done": False,
    "percent": 0,
    "stage": "idle",
    "message": "Ready",
    "result": None,
    "error": None,
}


def progress_snapshot() -> dict[str, Any]:
    with _sync_lock:
        return dict(_sync_state)


def set_progress(percent: int, stage: str, message: str) -> None:
    with _sync_lock:
        _sync_state["percent"] = max(0, min(100, int(percent)))
        _sync_state["stage"] = stage
        _sync_state["message"] = message


def require_gmail_authorized() -> None:
    if not gmail_client.is_authorized():
        raise AuthorizationError()


def build_sync_query() -> tuple[str, str]:
    base = f"in:inbox newer_than:{SYNC_WINDOW_DAYS}d"
    sync_state = storage.get_sync_state()
    if not sync_state.get("last_synced_at"):
        return base, "initial"
    latest = storage.latest_synced_email()
    if not latest or not latest.get("internal_date"):
        return base, "initial"
    received = datetime.fromtimestamp(int(latest["internal_date"]) / 1000, timezone.utc)
    date_str = received.strftime("%Y/%m/%d")
    return f"{base} after:{date_str}", "incremental"


def sync_inbox(*, analyze_new: bool = True, track_progress: bool = False) -> dict[str, Any]:
    require_gmail_authorized()
    query, mode = build_sync_query()
    new_count = 0
    fetched_count = 0
    errors: list[dict[str, str]] = []
    page_token: str | None = None
    size_estimate = 0

    def progress(percent: int, stage: str, message: str) -> None:
        if track_progress:
            set_progress(percent, stage, message)

    logger.info("Starting inbox sync mode=%s query=%s", mode, query)
    progress(2, "labels", "Preparing Gmail labels...")

    try:
        labels = gmail_client.ensure_job_labels()
        for label_name, label_id in labels.items():
            storage.save_label(label_name, label_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Label preparation failed: %s", exc)
        errors.append({"stage": "labels", "message": str(exc)})

    progress(8, "fetch", "Searching inbox (last 30 days)...")

    while fetched_count < SYNC_MAX_EMAILS:
        try:
            result = gmail_client.list_message_ids(
                query=query,
                max_results=SYNC_PAGE_SIZE,
                page_token=page_token,
            )
        except Exception as exc:
            logger.exception("Gmail search failed")
            raise ExternalServiceError(f"Gmail search failed: {exc}") from exc

        message_ids = result["ids"]
        fetched_count += len(message_ids)
        if not size_estimate:
            size_estimate = max(int(result.get("result_size_estimate") or 0), len(message_ids), 1)

        for message_id in message_ids:
            if storage.email_exists(message_id):
                continue
            try:
                message = gmail_client.get_message(message_id)
                storage.upsert_email(message)
                new_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to fetch message %s: %s", message_id, exc)
                errors.append({"stage": "message", "message_id": message_id, "message": str(exc)})

        fetch_percent = 8 + int(min(47, (fetched_count / size_estimate) * 47))
        progress(
            fetch_percent,
            "fetch",
            f"Fetching emails ({fetched_count} checked, {new_count} new)...",
        )

        page_token = result.get("next_page_token")
        if not page_token or not message_ids:
            break

    progress(55, "cleanup", "Saving sync state...")
    pruned = storage.prune_emails_older_than_days(SYNC_WINDOW_DAYS)
    latest = storage.latest_synced_email()
    if latest:
        storage.save_sync_state(latest.get("gmail_id"), latest.get("internal_date"))

    analyzed = 0
    pending_total = storage.count_pending_emails() if analyze_new else 0
    if analyze_new:
        progress(58, "analyze", f"Analyzing {pending_total} pending email(s)...")
        analyzed, analyze_errors = analyze_all_pending(
            track_progress=track_progress,
            total_pending=pending_total,
            on_progress=set_progress if track_progress else None,
        )
        errors.extend(analyze_errors[:10])

    progress(100, "done", "Sync complete")
    sync_info = storage.get_sync_info()
    result_payload = {
        "mode": mode,
        "query": query,
        "fetched": fetched_count,
        "new_emails": new_count,
        "analyzed": analyzed,
        "pruned": pruned,
        "pending_analysis": sync_info["pending_analysis"],
        "sync_info": sync_info,
        "errors": errors[:10],
    }
    logger.info(
        "Sync complete mode=%s fetched=%s new=%s analyzed=%s errors=%s",
        mode,
        fetched_count,
        new_count,
        analyzed,
        len(errors),
    )
    return result_payload


def start_background_sync(*, analyze_new: bool = True) -> dict[str, Any]:
    with _sync_lock:
        if _sync_state.get("active"):
            return {"started": False, "progress": dict(_sync_state)}
        _sync_state.update(
            {
                "active": True,
                "done": False,
                "percent": 0,
                "stage": "starting",
                "message": "Starting sync...",
                "result": None,
                "error": None,
            }
        )

    thread = threading.Thread(target=_run_sync_job, args=(analyze_new,), daemon=True)
    thread.start()
    return {"started": True, "progress": progress_snapshot()}


def _run_sync_job(analyze_new: bool) -> None:
    try:
        result = sync_inbox(analyze_new=analyze_new, track_progress=True)
        with _sync_lock:
            _sync_state.update(
                {
                    "active": False,
                    "done": True,
                    "percent": 100,
                    "stage": "done",
                    "message": "Sync complete",
                    "result": result,
                    "error": None,
                }
            )
    except Exception as exc:  # noqa: BLE001
        message = getattr(exc, "message", str(exc))
        logger.exception("Background sync failed")
        with _sync_lock:
            _sync_state.update(
                {
                    "active": False,
                    "done": True,
                    "percent": _sync_state.get("percent", 0),
                    "stage": "error",
                    "message": str(message),
                    "result": None,
                    "error": str(message),
                }
            )
