from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from friday.runtime.persist import atomic_write_json


def _empty_totals() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class TokenUsageStore:
    """Tracks Azure OpenAI token usage per call and aggregated scopes."""

    def __init__(
        self,
        *,
        call_log_max: int = 200,
        persist_path: Optional[Path] = None,
        persist_enabled: bool = True,
    ) -> None:
        self.call_log_max = max(10, call_log_max)
        self.persist_path = persist_path.resolve() if persist_path else None
        self.persist_enabled = persist_enabled and self.persist_path is not None
        self._call_log: Deque[Dict[str, Any]] = deque(maxlen=self.call_log_max)
        self._session = _empty_totals()
        self._session_by_source: Dict[str, Dict[str, int]] = {}
        self._last_turn = _empty_totals()
        self._last_turn_by_source: Dict[str, Dict[str, int]] = {}
        self._lifetime = _empty_totals()
        self._lifetime_by_source: Dict[str, Dict[str, int]] = {}
        if self.persist_enabled:
            self._load_lifetime()

    def _load_lifetime(self) -> None:
        if not self.persist_path or not self.persist_path.is_file():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        totals = data.get("lifetime") or {}
        if isinstance(totals, dict):
            self._lifetime = {
                "prompt_tokens": int(totals.get("prompt_tokens") or 0),
                "completion_tokens": int(totals.get("completion_tokens") or 0),
                "total_tokens": int(totals.get("total_tokens") or 0),
            }
        by_source = data.get("by_source") or {}
        if isinstance(by_source, dict):
            self._lifetime_by_source = {
                str(k): {
                    "prompt_tokens": int(v.get("prompt_tokens") or 0),
                    "completion_tokens": int(v.get("completion_tokens") or 0),
                    "total_tokens": int(v.get("total_tokens") or 0),
                }
                for k, v in by_source.items()
                if isinstance(v, dict)
            }

    def _save_lifetime(self) -> None:
        if not self.persist_enabled or not self.persist_path:
            return
        atomic_write_json(
            self.persist_path,
            {
                "lifetime": self._lifetime,
                "by_source": self._lifetime_by_source,
            },
        )

    @staticmethod
    def _normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        if not usage:
            return _empty_totals()
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    @staticmethod
    def _add_totals(target: Dict[str, int], delta: Dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            target[key] = target.get(key, 0) + delta.get(key, 0)

    @staticmethod
    def _add_by_source(
        bucket: Dict[str, Dict[str, int]],
        source: str,
        delta: Dict[str, int],
    ) -> None:
        entry = bucket.setdefault(source, _empty_totals())
        TokenUsageStore._add_totals(entry, delta)

    def start_turn(self) -> None:
        self._last_turn = _empty_totals()
        self._last_turn_by_source = {}

    def record(self, source: str, usage: Optional[Dict[str, Any]]) -> None:
        delta = self._normalize_usage(usage)
        if delta["total_tokens"] <= 0:
            return
        label = source.strip() or "unknown"
        self._call_log.append(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": label,
                **delta,
            }
        )
        self._add_totals(self._session, delta)
        self._add_by_source(self._session_by_source, label, delta)
        self._add_totals(self._last_turn, delta)
        self._add_by_source(self._last_turn_by_source, label, delta)
        self._add_totals(self._lifetime, delta)
        self._add_by_source(self._lifetime_by_source, label, delta)
        self._save_lifetime()

    def snapshot(self, scope: str = "session") -> Dict[str, Any]:
        scope_key = scope.strip().lower()
        if scope_key == "last_turn":
            totals = dict(self._last_turn)
            by_source = {k: dict(v) for k, v in self._last_turn_by_source.items()}
        elif scope_key == "lifetime":
            totals = dict(self._lifetime)
            by_source = {k: dict(v) for k, v in self._lifetime_by_source.items()}
        else:
            totals = dict(self._session)
            by_source = {k: dict(v) for k, v in self._session_by_source.items()}
        return {
            "scope": scope_key if scope_key in {"last_turn", "lifetime"} else "session",
            "totals": totals,
            "by_source": by_source,
            "recent_calls": list(self._call_log)[-20:],
        }
