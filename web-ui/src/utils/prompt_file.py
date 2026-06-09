"""Load/save the default agent task prompt (web-ui/prompt.txt)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_WEBUI_ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = _WEBUI_ROOT / "prompt.txt"


def load_default_prompt() -> str:
    """Return prompt.txt contents, or empty string if missing."""
    try:
        if PROMPT_FILE.is_file():
            return PROMPT_FILE.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not read %s: %s", PROMPT_FILE, exc)
    return ""


def save_default_prompt(text: str) -> None:
    """Persist task prompt to prompt.txt (auto-save on run)."""
    try:
        PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROMPT_FILE.write_text(text, encoding="utf-8")
        logger.info("Saved agent prompt to %s", PROMPT_FILE)
    except Exception as exc:
        logger.warning("Could not save %s: %s", PROMPT_FILE, exc)
        raise
