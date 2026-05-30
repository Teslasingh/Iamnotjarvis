"""Heuristic analysis of captured shell stdout/stderr (inspired by robust CLI runners)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Patterns that usually indicate Python / script failure even when tooling misreports exit codes.
_STRONG_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("traceback", r"(?im)^traceback\b|traceback\s*\("),
    ("syntax_error", r"\bsyntaxerror\b"),
    ("indentation_error", r"\bindentationerror\b"),
    ("import_error", r"\bimporterror\b"),
    ("module_not_found", r"\bmodulenotfounderror\b"),
    ("fatal_error", r"\bfatal(\s+error)?\b"),
    ("assertion_failed", r"\bassertionerror\b"),
    ("process_exited", r"exited with code\s*[1-9]"),
    ("command_failed", r"command failed"),
)

_WEAK_PATTERNS: Tuple[Tuple[str, str, int], ...] = (
    ("exception_line", r"^\s*\w+error\s*:", re.MULTILINE | re.IGNORECASE),
    ("python_exception", r"\b(type|value|name|key|index|attribute)error\s*:", re.IGNORECASE),
)


def analyze_shell_streams(
    stdout: str,
    stderr: str,
    return_code: Optional[int],
) -> Dict[str, Any]:
    """Summarize whether output looks like a failure; never replaces the real return code."""
    out = stdout or ""
    err = stderr or ""
    combined = f"{out}\n{err}"

    signals: List[str] = []
    for name, pattern in _STRONG_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            signals.append(name)
    for name, pattern, flags in _WEAK_PATTERNS:
        if re.search(pattern, combined, flags):
            signals.append(name)

    seen: set[str] = set()
    ordered: List[str] = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    stderr_nonempty = bool(err.strip())
    rc = return_code if return_code is not None else -1
    exit_ok = rc == 0

    suspect = False
    if not exit_ok:
        suspect = True
    elif stderr_nonempty and re.search(r"error|exception|traceback", err, re.IGNORECASE):
        suspect = True
    elif any(s in ordered for s in ("traceback", "syntax_error", "fatal_error")):
        suspect = True

    parts: List[str] = []
    if exit_ok:
        parts.append("exit 0")
    else:
        parts.append(f"exit {rc}")
    if ordered:
        parts.append("signals: " + ", ".join(ordered[:6]))
    elif stderr_nonempty and exit_ok:
        parts.append("stderr non-empty")
    summary = " · ".join(parts) if parts else "no output"

    return {
        "exit_ok": exit_ok,
        "return_code": return_code,
        "execution_signals": ordered[:12],
        "stderr_nonempty": stderr_nonempty,
        "suspect_failure": suspect,
        "summary": summary,
    }
