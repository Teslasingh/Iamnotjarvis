"""Draft creation and sending."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import ai_agent
import gmail_client
import prompt_store
import storage
from errors import NotFoundError, ValidationError
from profile_store import load_persisted_profile, resume_path
from services.analysis_service import apply_gmail_status_label
from services.sync_service import require_gmail_authorized

logger = logging.getLogger(__name__)


def create_draft(gmail_id: str) -> dict[str, Any]:
    email = storage.get_email(gmail_id)
    if not email:
        raise NotFoundError("Email not found")

    profile = load_persisted_profile()
    agent_prompt = prompt_store.load_agent_prompt()
    analysis = email.get("analysis_json") or ai_agent.analyze_email(
        email,
        profile,
        agent_prompt=agent_prompt,
    )
    draft_data = ai_agent.draft_reply(email, analysis, profile, agent_prompt=agent_prompt)
    saved = storage.save_draft(
        gmail_id,
        draft_data["subject"],
        draft_data["body"],
        bool(draft_data.get("attach_resume")),
    )
    logger.info("Created draft %s for email %s", saved.get("id"), gmail_id)
    return saved


def update_draft(draft_id: int, *, subject: str, body: str, attach_resume: bool) -> dict[str, Any]:
    draft_data = storage.update_draft(draft_id, subject, body, attach_resume)
    if not draft_data:
        raise NotFoundError("Draft not found or already sent")
    return draft_data


def send_draft(draft_id: int) -> dict[str, Any]:
    require_gmail_authorized()
    draft_data = storage.get_draft(draft_id)
    if not draft_data:
        raise NotFoundError("Draft not found")
    if draft_data.get("status") == "sent":
        raise ValidationError("Draft was already sent")

    email = storage.get_email(draft_data["gmail_id"])
    if not email:
        raise NotFoundError("Email not found")

    attachments: list[Path] = []
    profile = load_persisted_profile()
    current_resume = resume_path(profile)
    if draft_data.get("attach_resume"):
        if not current_resume:
            raise ValidationError("Draft requires a resume, but no resume is uploaded")
        attachments.append(current_resume)

    try:
        sent = gmail_client.send_message(
            to_address=email.get("sender_email") or email.get("sender") or "",
            subject=draft_data["subject"],
            body=draft_data["body"],
            thread_id=email.get("thread_id"),
            attachments=attachments,
        )
    except Exception as exc:
        logger.exception("Failed to send draft %s", draft_id)
        from errors import ExternalServiceError

        raise ExternalServiceError(f"Failed to send message: {exc}") from exc

    storage.mark_draft_sent(draft_id, sent.get("id", ""))
    storage.update_email_status(email["gmail_id"], "Relevant")
    apply_gmail_status_label(email["gmail_id"], "Relevant")
    logger.info("Sent draft %s as Gmail message %s", draft_id, sent.get("id"))
    return {"sent": True, "message": sent, "draft": storage.get_draft(draft_id)}
