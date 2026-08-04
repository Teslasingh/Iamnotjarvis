"""Application logging configuration."""

from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False

# Endpoints the browser polls on a tight interval. Their access-log lines are
# pure noise in the terminal, so we drop them while keeping real requests.
_SUPPRESSED_ACCESS_PATHS = ("/api/sync/progress",)


class _SuppressPollingAccessFilter(logging.Filter):
    """Drop uvicorn access-log lines for high-frequency polling endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True
        message = record.getMessage()
        return not any(path in message for path in _SUPPRESSED_ACCESS_PATHS)


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once for the process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("EMAIL_AGENT_LOG_LEVEL") or "INFO").upper()
    numeric = getattr(logging, resolved, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(_SuppressPollingAccessFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)

    # Uvicorn installs its own handlers (propagate=False) before our lifespan
    # runs, so attach the same filter there to silence polling access logs.
    for uvicorn_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        for uvicorn_handler in uvicorn_logger.handlers:
            uvicorn_handler.addFilter(_SuppressPollingAccessFilter())

    # Quiet noisy third-party loggers unless debugging.
    if numeric > logging.DEBUG:
        logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured at %s", resolved)
