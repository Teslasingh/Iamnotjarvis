from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TurnMistakeLog:
    """Collects failures during an agent turn for long-term soul learning."""

    entries: List[str] = field(default_factory=list)
    step_budget_exhausted: bool = False

    def record(self, summary: str) -> None:
        text = " ".join(summary.split()).strip()
        if not text:
            return
        if len(text) > 500:
            text = text[:497].rstrip() + "..."
        if self.entries and self.entries[-1] == text:
            return
        self.entries.append(text)

    def merge(self, other: "TurnMistakeLog") -> None:
        for entry in other.entries:
            self.record(entry)

    def has_entries(self) -> bool:
        return bool(self.entries)

    def format_for_soul(self, max_chars: int = 4000) -> str:
        if not self.entries:
            return ""
        lines = [f"- {entry}" for entry in self.entries]
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20].rstrip() + "\n[... truncated ...]"


def shell_run_failed(result: str) -> bool:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return False
    if payload.get("error"):
        return True
    outcome = payload.get("outcome") or {}
    return bool(outcome.get("suspect_failure")) or outcome.get("exit_ok") is False


def extract_shell_mistake(result: str) -> Optional[str]:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return None
    if payload.get("error") == "repeated_tool_call":
        return str(payload.get("message") or "repeated tool call")
    if payload.get("error"):
        cmd = str(payload.get("command") or payload.get("cmd") or "shell command").strip()
        return f"shell error ({cmd}): {payload['error']}"
    if not shell_run_failed(result):
        return None
    cmd = str(payload.get("command") or payload.get("cmd") or "shell command").strip()
    outcome = payload.get("outcome") or {}
    summary = str(outcome.get("summary") or "").strip()
    stderr = str(payload.get("stderr") or "").strip()
    detail = summary or stderr[:240] or f"exit {payload.get('return_code', '?')}"
    return f"shell failed ({cmd}): {detail}"


def extract_tool_mistake(name: str, result: str) -> Optional[str]:
    if name in {"run_shell", "start_shell_job", "get_shell_job"}:
        return extract_shell_mistake(result)
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("error"):
        return f"{name} error: {payload['error']}"
    if name == "validate_python" and payload.get("ok") is False:
        failed = [
            item
            for item in (payload.get("results") or [])
            if isinstance(item, dict) and not item.get("ok")
        ]
        if failed:
            item = failed[0]
            path = item.get("path") or "unknown file"
            err = str(item.get("stderr") or item.get("error") or "syntax error").strip()[:200]
            return f"validate_python failed ({path}): {err}"
    return None


def record_tool_outcome(log: TurnMistakeLog, name: str, args: Dict[str, Any], result: str) -> None:
    mistake = extract_tool_mistake(name, result)
    if mistake:
        log.record(mistake)
        return
    if name in {"run_shell", "get_shell_job"} and shell_run_failed(result):
        fallback = extract_shell_mistake(result)
        if fallback:
            log.record(fallback)
