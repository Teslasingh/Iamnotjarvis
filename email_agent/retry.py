"""Lightweight retry helpers for transient external failures."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_transient_exception(exc: BaseException) -> bool:
    """Best-effort detection of retryable HTTP / network errors."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in TRANSIENT_STATUS_CODES:
        return True

    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status in TRANSIENT_STATUS_CODES:
            return True

    # googleapiclient.errors.HttpError exposes .resp.status
    resp = getattr(exc, "resp", None)
    if resp is not None:
        resp_status = getattr(resp, "status", None)
        try:
            if int(resp_status) in TRANSIENT_STATUS_CODES:
                return True
        except (TypeError, ValueError):
            pass

    message = str(exc).lower()
    markers = ("rate limit", "timeout", "temporarily unavailable", "connection reset", "503", "429")
    return any(marker in message for marker in markers)


def with_retry(
    operation: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.6,
    max_delay: float = 8.0,
    retry_on: Callable[[BaseException], bool] | None = None,
    operation_name: str = "operation",
) -> T:
    """Run ``operation`` with exponential backoff + jitter on transient errors."""
    should_retry = retry_on or is_transient_exception
    last_error: BaseException | None = None

    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 — intentional boundary
            last_error = exc
            if attempt >= attempts or not should_retry(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)
            logger.warning(
                "%s failed (attempt %s/%s): %s; retrying in %.2fs",
                operation_name,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error
