"""Detect when the user wants host execution (run_shell) vs pure Q&A."""

from __future__ import annotations

import re

_EXECUTION_PHRASES = (
    "on my system",
    "on this system",
    "on my machine",
    "on this machine",
    "on my computer",
    "on the host",
    "on this host",
    "run it",
    "run this",
    "execute it",
    "execute this",
    "my system",
    "this system",
)

_STATUS_QUERY_PATTERNS = (
    re.compile(r"\bwhat(?:'s| is| are)\s+.+\brunning\b", re.I),
    re.compile(r"\b(list|show|check|get)\s+.+\brunning\b", re.I),
    re.compile(r"\btasks?\s+running\b", re.I),
    re.compile(r"\bprocesses?\s+running\b", re.I),
    re.compile(r"\brunning\s+in\s+(tmux|docker|podman|kubernetes|k8s)\b", re.I),
    re.compile(r"\b(tmux|docker|kubectl)\s+(ls|ps|list)\b", re.I),
)

_BARE_COMMAND = re.compile(
    r"^(?:run\s+)?(tmux|docker|kubectl|npm|pip|python|git|systemctl|ps|ls|dir|where|winget|choco)\b",
    re.I,
)


def implies_host_execution(message: str) -> bool:
    """True when the user expects live host inspection via shell tools, not a tutorial."""
    text = str(message or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in _EXECUTION_PHRASES):
        return True
    if re.search(r"\b(run|execute)\b", lowered):
        return True
    for pattern in _STATUS_QUERY_PATTERNS:
        if pattern.search(text):
            return True
    if _BARE_COMMAND.match(text):
        return True
    return False
