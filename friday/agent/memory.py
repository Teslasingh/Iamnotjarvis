from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class MemoryStore:
    """Session-scoped in-process conversation memory."""

    def __init__(self, recent_turns: int = 12) -> None:
        self.recent_turns = max(1, recent_turns)
        self._records: List[Dict[str, Any]] = []

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

    def clear(self) -> None:
        self._records.clear()

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
        parts = []
        for idx, turn in enumerate(turns, 1):
            user = str(turn.get("user", ""))[:1200]
            assistant = str(turn.get("assistant", ""))[:1600]
            att_lines = []
            for item in turn.get("attachments") or []:
                path = item.get("path") or item.get("name")
                if path:
                    att_lines.append(f"  attachment: {path}")
            att_block = ("\n" + "\n".join(att_lines)) if att_lines else ""
            parts.append(f"Turn {idx}\nUser: {user}{att_block}\nAssistant: {assistant}")
        return "\n\n".join(parts)
