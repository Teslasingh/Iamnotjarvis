from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from friday.runtime.persist import atomic_write_json

logger = logging.getLogger(__name__)


class MemoryStore:
    """Conversation memory with optional disk persistence."""

    def __init__(
        self,
        recent_turns: int = 12,
        *,
        persist_path: Optional[Path] = None,
        max_context_chars: int = 12000,
        persist_enabled: bool = True,
    ) -> None:
        self.recent_turns = max(1, recent_turns)
        self.max_context_chars = max(500, max_context_chars)
        self.persist_path = persist_path.resolve() if persist_path else None
        self.persist_enabled = persist_enabled and self.persist_path is not None
        self._records: List[Dict[str, Any]] = []
        if self.persist_enabled:
            self.load()

    def load(self) -> None:
        if not self.persist_path or not self.persist_path.is_file():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        records = data.get("turns")
        if isinstance(records, list):
            self._records = records[-self.recent_turns :]

    def _save(self) -> None:
        if not self.persist_enabled or not self.persist_path:
            return
        try:
            atomic_write_json(self.persist_path, {"turns": self._records})
        except OSError as exc:
            logger.warning("conversation memory persist failed: %s", exc)

    def append_turn(
        self,
        user_message: str,
        assistant_reply: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": user_message,
            "assistant": assistant_reply,
            "attachments": attachments or [],
        }
        self._records.append(record)
        self.compact()
        self._save()

    def clear(self) -> None:
        self._records.clear()
        self._save()

    def compact(self) -> None:
        self._records = self._records[-self.recent_turns :]

    def recent_turns_list(self) -> List[Dict[str, Any]]:
        return list(self._records[-self.recent_turns :])

    def recent_attachments(self) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        for turn in self._records:
            for item in turn.get("attachments") or []:
                path = str(item.get("path") or "")
                if path:
                    seen[path] = item
        return list(seen.values())

    def build_context(self) -> str:
        turns = self.recent_turns_list()
        if not turns:
            return ""
        parts: List[str] = []
        total = 0
        for idx, turn in enumerate(turns, 1):
            user = str(turn.get("user", ""))[:1200]
            assistant = str(turn.get("assistant", ""))[:1600]
            att_lines = []
            for item in turn.get("attachments") or []:
                path = item.get("path") or item.get("name")
                if path:
                    att_lines.append(f"  attachment: {path}")
            att_block = ("\n" + "\n".join(att_lines)) if att_lines else ""
            block = f"Turn {idx}\nUser: {user}{att_block}\nAssistant: {assistant}"
            if total and total + len(block) + 2 > self.max_context_chars:
                break
            parts.append(block)
            total += len(block) + 2
        return "\n\n".join(parts)
