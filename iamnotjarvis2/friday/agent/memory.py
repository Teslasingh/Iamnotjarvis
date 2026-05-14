from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List


class MemoryStore:
    """Session-scoped local conversation memory.

    Stores concise chat turns only for the currently running app session.
    The web app clears this file on startup and shutdown so history does not
    accumulate across runs.
    """

    def __init__(self, memory_dir: str, recent_turns: int = 12) -> None:
        self.memory_dir = memory_dir
        self.recent_turns = max(1, recent_turns)
        os.makedirs(self.memory_dir, exist_ok=True)
        self.path = os.path.join(self.memory_dir, "conversation.jsonl")

    def append_turn(self, user_message: str, assistant_reply: str) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": user_message,
            "assistant": assistant_reply,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.compact()

    def clear(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8"):
                pass
        except OSError:
            pass

    def compact(self) -> None:
        records = self.recent_turns_list()
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def recent_turns_list(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return records[-self.recent_turns :]

    def build_context(self) -> str:
        turns = self.recent_turns_list()
        if not turns:
            return ""
        parts = []
        for idx, turn in enumerate(turns, 1):
            user = str(turn.get("user", ""))[:1200]
            assistant = str(turn.get("assistant", ""))[:1600]
            parts.append(f"Turn {idx}\nUser: {user}\nAssistant: {assistant}")
        return "\n\n".join(parts)
