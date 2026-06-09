"""
Cross-task session memory for job-application runs (no extra vector DB).

Persists applied/skipped jobs, recent agent memory lines, and URLs so the LLM
does not repeat work when the browser session stays open.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_WEBUI_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SESSION_FILE = _WEBUI_ROOT / "tmp" / "webui_session" / "job_session.json"

_APPLIED_RE = re.compile(
    r"\b(applied|application submitted|successfully submitted|easy apply complete)\b",
    re.I,
)
_SKIP_RE = re.compile(
    r"\b(skip(?:ped)?|already applied|not relevant|external apply|too long)\b",
    re.I,
)
_LOGIN_RE = re.compile(r"\b(logged in|login success|signed in)\b", re.I)


@dataclass
class JobSessionContext:
    """Rolling session state across agent steps and task runs."""

    tasks_started: List[str] = field(default_factory=list)
    jobs_applied: List[dict[str, str]] = field(default_factory=list)
    jobs_skipped: List[dict[str, str]] = field(default_factory=list)
    recent_urls: List[str] = field(default_factory=list)
    agent_memories: List[str] = field(default_factory=list)
    user_learnings: List[dict[str, str]] = field(default_factory=list)
    last_evaluation: str = ""
    last_next_goal: str = ""
    login_status: str = ""
    updated_at: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "JobSessionContext":
        path = path or _session_file_path()
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as exc:
            logger.warning("Could not load session context from %s: %s", path, exc)
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        path = path or _session_file_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.updated_at = datetime.now(timezone.utc).isoformat()
            path.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save session context: %s", exc)

    @classmethod
    def clear_persisted(cls, path: Optional[Path] = None) -> None:
        path = path or _session_file_path()
        try:
            if path.is_file():
                path.unlink()
        except Exception as exc:
            logger.warning("Could not clear session file: %s", exc)

    def reset(self) -> None:
        self.tasks_started.clear()
        self.jobs_applied.clear()
        self.jobs_skipped.clear()
        self.recent_urls.clear()
        self.agent_memories.clear()
        self.user_learnings.clear()
        self.last_evaluation = ""
        self.last_next_goal = ""
        self.login_status = ""
        self.updated_at = ""

    def record_user_learning(
        self,
        message: str,
        *,
        url: str = "",
        learning_type: str = "",
    ) -> None:
        msg = (message or "").strip()[:500]
        if not msg:
            return
        entry = {
            "message": msg,
            "url": (url or "")[:200],
            "type": (learning_type or "")[:40],
        }
        for existing in self.user_learnings[-6:]:
            if existing.get("message") == entry["message"]:
                return
        self.user_learnings.append(entry)
        self.user_learnings = self.user_learnings[-20:]

    def record_task_start(self, task: str) -> None:
        snippet = (task or "").strip().replace("\n", " ")[:240]
        if snippet and (not self.tasks_started or self.tasks_started[-1] != snippet):
            self.tasks_started.append(snippet)
            self.tasks_started = self.tasks_started[-6:]

    def update_from_step(
        self,
        *,
        url: str = "",
        title: str = "",
        memory: str = "",
        evaluation: str = "",
        next_goal: str = "",
        result_snippets: Optional[List[str]] = None,
    ) -> None:
        if url:
            u = url[:300]
            if not self.recent_urls or self.recent_urls[-1] != u:
                self.recent_urls.append(u)
                self.recent_urls = self.recent_urls[-15:]

        if memory:
            m = memory.strip()[:600]
            if not self.agent_memories or self.agent_memories[-1] != m:
                self.agent_memories.append(m)
                self.agent_memories = self.agent_memories[-10:]

        if evaluation:
            self.last_evaluation = evaluation.strip()[:400]
        if next_goal:
            self.last_next_goal = next_goal.strip()[:400]

        blob = " ".join(
            [memory, evaluation, next_goal, title]
            + (result_snippets or [])
        )
        entry = {
            "url": url[:200],
            "title": (title or "")[:120],
            "note": (memory or evaluation or next_goal)[:200],
        }

        if _LOGIN_RE.search(blob) and not self.login_status:
            self.login_status = "logged_in"

        if _APPLIED_RE.search(blob):
            if not _already_tracked(self.jobs_applied, entry):
                self.jobs_applied.append(entry)
                self.jobs_applied = self.jobs_applied[-25:]

        if _SKIP_RE.search(blob):
            if not _already_tracked(self.jobs_skipped, entry):
                self.jobs_skipped.append(entry)
                self.jobs_skipped = self.jobs_skipped[-25:]

    def format_injection(self, *, step_number: int = 0, inject_interval: int = 1) -> Optional[str]:
        """Build LLM context block; throttle repeats on early steps."""
        if inject_interval > 1 and step_number > 0 and step_number % inject_interval != 0:
            if step_number != 1:
                return None

        if not any(
            [
                self.tasks_started,
                self.jobs_applied,
                self.jobs_skipped,
                self.agent_memories,
                self.user_learnings,
                self.recent_urls,
                self.login_status,
            ]
        ):
            return None

        lines = [
            "[Session context — use this with the current page; do not repeat finished work]",
        ]
        if self.login_status:
            lines.append(f"- Login: {self.login_status}")
        if self.tasks_started:
            lines.append(f"- Tasks this session: {len(self.tasks_started)} (latest: {self.tasks_started[-1][:160]}…)")
        if self.jobs_applied:
            lines.append(f"- Already applied ({len(self.jobs_applied)}):")
            for j in self.jobs_applied[-5:]:
                lines.append(f"  • {j.get('title') or j.get('url', '?')[:80]}")
        if self.jobs_skipped:
            lines.append(f"- Skipped ({len(self.jobs_skipped)}):")
            for j in self.jobs_skipped[-5:]:
                lines.append(f"  • {j.get('title') or j.get('url', '?')[:80]}: {j.get('note', '')[:80]}")
        if self.agent_memories:
            lines.append("- Recent agent memory:")
            for m in self.agent_memories[-4:]:
                lines.append(f"  • {m[:220]}")
        if self.user_learnings:
            lines.append("- User corrections (follow these; see also [User preferences] when injected):")
            for item in self.user_learnings[-6:]:
                lines.append(f"  • {item.get('message', '')[:220]}")
        if self.last_next_goal:
            lines.append(f"- Last planned goal: {self.last_next_goal[:200]}")
        if self.recent_urls:
            lines.append(
                "- Recent URLs: " + " | ".join(self.recent_urls[-3:])
            )
        lines.append(
            "- Update your `memory` field each step with counts (e.g. applied X, skipped Y, current job URL)."
        )
        return "\n".join(lines)


def _already_tracked(items: List[dict[str, str]], entry: dict[str, str]) -> bool:
    key = entry.get("url") or entry.get("title")
    if not key:
        return False
    for existing in items[-8:]:
        if (existing.get("url") and existing.get("url") == entry.get("url")) or (
            existing.get("title") and existing.get("title") == entry.get("title")
        ):
            return True
    return False


def _session_file_path() -> Path:
    custom = os.getenv("JOB_SESSION_FILE", "").strip()
    if custom:
        return Path(custom)
    return _DEFAULT_SESSION_FILE


def session_enabled() -> bool:
    return os.getenv("JOB_SESSION_CONTEXT", "true").lower() in ("1", "true", "yes", "on")


def session_inject_interval() -> int:
    try:
        return max(1, int(os.getenv("JOB_SESSION_INJECT_INTERVAL", "1")))
    except ValueError:
        return 1
