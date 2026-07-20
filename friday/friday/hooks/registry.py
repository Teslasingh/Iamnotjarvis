from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from friday.runtime.persist import atomic_write_json


class HookRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._hooks: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            self._hooks = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._hooks = list(data) if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            self._hooks = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self._hooks)

    def list_hooks(self) -> List[Dict[str, Any]]:
        return list(self._hooks)

    def add(self, hook: Dict[str, Any]) -> Dict[str, Any]:
        self._hooks.append(hook)
        self.save()
        return hook

    def delete(self, hook_id: str) -> bool:
        before = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.get("id") != hook_id]
        if len(self._hooks) < before:
            self.save()
            return True
        return False
