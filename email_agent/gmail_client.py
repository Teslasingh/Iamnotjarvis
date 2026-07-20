"""Gmail OAuth and API client with retries and structured logging."""

from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from config import CLIENT_SECRET_FILE, OAUTH_STATE_FILE, TOKEN_FILE, ensure_directories, settings
from constants import DEFAULT_JOB_LABEL_STATUS, JOB_LABELS
from errors import AuthorizationError, ValidationError
from retry import with_retry

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.send",
]

# Re-export for callers that import JOB_LABELS from this module.
__all__ = [
    "JOB_LABELS",
    "SCOPES",
    "apply_status_label",
    "archive_message",
    "authorization_url",
    "ensure_job_labels",
    "fetch_token",
    "get_message",
    "gmail_profile",
    "is_authorized",
    "list_message_ids",
    "parse_message",
    "send_message",
    "service",
]


def _load_credentials() -> Credentials | None:
    ensure_directories()
    if not TOKEN_FILE.exists():
        return None
    try:
        credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load Gmail credentials: %s", exc)
        return None

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            with_retry(
                lambda: credentials.refresh(Request()),
                attempts=3,
                operation_name="gmail.token_refresh",
            )
            TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gmail token refresh failed: %s", exc)
            return None
    return credentials if credentials and credentials.valid else None


def is_authorized() -> bool:
    return _load_credentials() is not None


def authorization_url() -> str:
    _allow_local_http_oauth()
    flow = Flow.from_client_secrets_file(str(CLIENT_SECRET_FILE), scopes=SCOPES)
    flow.redirect_uri = settings.oauth_redirect_uri
    url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    _save_oauth_state(state=state, code_verifier=getattr(flow, "code_verifier", None))
    logger.info("Started Gmail OAuth flow")
    return url


def fetch_token(authorization_response: str) -> None:
    ensure_directories()
    _allow_local_http_oauth()
    oauth_state = _load_oauth_state()
    expected_state = oauth_state.get("state") or ""
    callback_state = _extract_query_param(authorization_response, "state")
    if expected_state and callback_state != expected_state:
        raise ValidationError("OAuth state mismatch. Restart Connect Gmail.")

    flow = Flow.from_client_secrets_file(str(CLIENT_SECRET_FILE), scopes=SCOPES)
    flow.redirect_uri = settings.oauth_redirect_uri
    if oauth_state.get("code_verifier"):
        flow.code_verifier = oauth_state["code_verifier"]
    flow.fetch_token(authorization_response=authorization_response)
    TOKEN_FILE.write_text(flow.credentials.to_json(), encoding="utf-8")
    OAUTH_STATE_FILE.unlink(missing_ok=True)
    logger.info("Gmail OAuth token stored")


def service():
    credentials = _load_credentials()
    if not credentials:
        raise AuthorizationError()
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def _allow_local_http_oauth() -> None:
    redirect = urlparse(settings.oauth_redirect_uri)
    if redirect.scheme == "http" and redirect.hostname in {"127.0.0.1", "localhost"}:
        os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")


def _save_oauth_state(state: str | None, code_verifier: str | None) -> None:
    ensure_directories()
    OAUTH_STATE_FILE.write_text(
        json.dumps({"state": state or "", "code_verifier": code_verifier or ""}),
        encoding="utf-8",
    )


def _load_oauth_state() -> dict[str, str]:
    if not OAUTH_STATE_FILE.exists():
        return {}
    try:
        data = json.loads(OAUTH_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        "state": str(data.get("state") or ""),
        "code_verifier": str(data.get("code_verifier") or ""),
    }


def _extract_query_param(url: str, name: str) -> str:
    query = parse_qs(urlparse(url).query)
    values = query.get(name) or []
    return values[0] if values else ""


def _execute(request: Any, *, operation_name: str) -> Any:
    return with_retry(request.execute, attempts=3, operation_name=operation_name)


def gmail_profile() -> dict[str, Any]:
    return _execute(service().users().getProfile(userId="me"), operation_name="gmail.profile")


def list_message_ids(query: str, max_results: int = 25, page_token: str | None = None) -> dict[str, Any]:
    request = (
        service()
        .users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results, pageToken=page_token)
    )
    response = _execute(request, operation_name="gmail.list_messages")
    return {
        "ids": [item["id"] for item in response.get("messages", [])],
        "next_page_token": response.get("nextPageToken"),
        "result_size_estimate": response.get("resultSizeEstimate", 0),
    }


def get_message(message_id: str) -> dict[str, Any]:
    request = service().users().messages().get(userId="me", id=message_id, format="full")
    raw = _execute(request, operation_name="gmail.get_message")
    return parse_message(raw)


def ensure_job_labels() -> dict[str, str]:
    existing = _execute(
        service().users().labels().list(userId="me"),
        operation_name="gmail.list_labels",
    ).get("labels", [])
    by_name = {label["name"]: label["id"] for label in existing}
    resolved: dict[str, str] = {}
    for label_name in JOB_LABELS.values():
        if label_name not in by_name:
            created = _execute(
                service()
                .users()
                .labels()
                .create(
                    userId="me",
                    body={
                        "name": label_name,
                        "labelListVisibility": "labelShow",
                        "messageListVisibility": "show",
                    },
                ),
                operation_name="gmail.create_label",
            )
            by_name[label_name] = created["id"]
            logger.info("Created Gmail label %s", label_name)
        resolved[label_name] = by_name[label_name]
    return resolved


def apply_status_label(message_id: str, status: str) -> str:
    labels = ensure_job_labels()
    label_status = status if status in JOB_LABELS else DEFAULT_JOB_LABEL_STATUS
    label_name = JOB_LABELS[label_status]
    label_id = labels[label_name]
    remove_label_ids = [current_id for name, current_id in labels.items() if name != label_name]
    _execute(
        service()
        .users()
        .messages()
        .modify(
            userId="me",
            id=message_id,
            body={"addLabelIds": [label_id], "removeLabelIds": remove_label_ids},
        ),
        operation_name="gmail.apply_label",
    )
    return label_id


def archive_message(message_id: str) -> None:
    _execute(
        service()
        .users()
        .messages()
        .modify(userId="me", id=message_id, body={"removeLabelIds": ["INBOX"]}),
        operation_name="gmail.archive",
    )


def send_message(
    to_address: str,
    subject: str,
    body: str,
    thread_id: str | None = None,
    attachments: list[Path] | None = None,
) -> dict[str, Any]:
    message = _build_message(to_address, subject, body, thread_id=thread_id, attachments=attachments or [])
    return _execute(
        service().users().messages().send(userId="me", body=message),
        operation_name="gmail.send",
    )


def parse_message(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload", {})
    headers = {item["name"].lower(): item.get("value", "") for item in payload.get("headers", [])}
    sender = headers.get("from", "")
    sender_email = getaddresses([sender])[0][1] if sender else ""
    internal_date = int(raw.get("internalDate", "0") or 0)
    received_at = _received_at(headers.get("date"), internal_date)
    return {
        "id": raw["id"],
        "thread_id": raw.get("threadId", ""),
        "history_id": raw.get("historyId"),
        "sender": sender,
        "sender_email": sender_email,
        "recipient": headers.get("to", ""),
        "subject": headers.get("subject", "(no subject)"),
        "snippet": raw.get("snippet", ""),
        **_extract_bodies(payload),
        "received_at": received_at,
        "internal_date": internal_date,
        "label_ids": raw.get("labelIds", []),
    }


def _received_at(date_header: str | None, internal_date: int) -> str:
    if date_header:
        try:
            parsed = parsedate_to_datetime(date_header)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except (TypeError, ValueError):
            pass
    if internal_date:
        return datetime.fromtimestamp(internal_date / 1000, timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _extract_bodies(payload: dict[str, Any]) -> dict[str, str]:
    parts = _walk_parts(payload)
    plain = [part for mime, part in parts if mime == "text/plain" and part.strip()]
    html_parts = [part for mime, part in parts if mime == "text/html" and part.strip()]
    body_html = "\n".join(html_parts).strip()
    plain_text = "\n\n".join(plain).strip()
    html_as_text = _html_to_text(body_html) if body_html else ""

    if plain_text and html_as_text:
        body_text = plain_text if len(plain_text) >= max(80, int(len(html_as_text) * 0.45)) else html_as_text
    elif plain_text:
        body_text = plain_text
    elif html_as_text:
        body_text = html_as_text
    else:
        body = payload.get("body", {}).get("data")
        body_text = _decode_body(body) if body else ""

    return {"body_text": body_text.strip(), "body_html": body_html}


def _walk_parts(payload: dict[str, Any]) -> list[tuple[str, str]]:
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")
    results: list[tuple[str, str]] = []
    if body_data:
        results.append((mime_type, _decode_body(body_data)))
    for part in payload.get("parts", []) or []:
        results.extend(_walk_parts(part))
    return results


def _decode_body(data: str) -> str:
    decoded = base64.urlsafe_b64decode(data.encode("utf-8"))
    return decoded.decode("utf-8", errors="replace")


def _html_to_text(value: str) -> str:
    value = re.sub(r"(?is)<(script|style|noscript|head).*?>.*?</\1>", " ", value)
    value = re.sub(r"(?is)<!--.*?-->", " ", value)
    value = re.sub(
        r'(?is)<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        lambda match: f"{_strip_tags(match.group(2)).strip()} ({match.group(1).strip()})",
        value,
    )
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)<li(\s[^>]*)?>", "\n• ", value)
    value = re.sub(r"(?i)</(p|div|tr|h[1-6]|li|blockquote)>", "\n", value)
    value = re.sub(r"(?i)</(table|ul|ol|hr)>", "\n\n", value)
    value = re.sub(r"(?i)<(p|div|tr|h[1-6]|blockquote|br)(\s[^>]*)?>", "\n", value)
    value = re.sub(r"(?i)<hr\s*/?>", "\n---\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "")


def _build_message(
    to_address: str,
    subject: str,
    body: str,
    thread_id: str | None = None,
    attachments: list[Path] | None = None,
) -> dict[str, Any]:
    email = EmailMessage()
    email["To"] = to_address
    email["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    email.set_content(body)

    for attachment in attachments or []:
        data = attachment.read_bytes()
        email.add_attachment(
            data,
            maintype="application",
            subtype="octet-stream",
            filename=attachment.name,
        )

    encoded = base64.urlsafe_b64encode(email.as_bytes()).decode("utf-8")
    message: dict[str, Any] = {"raw": encoded}
    if thread_id:
        message["threadId"] = thread_id
    return message
