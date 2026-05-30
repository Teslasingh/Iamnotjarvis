from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Set


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class EventBus:
    """In-memory pub/sub with a bounded ring of recent events for late subscribers."""

    ring_max: int = 500
    _subscribers: Set[asyncio.Queue] = field(default_factory=set, repr=False)
    _ring: Deque[Dict[str, Any]] = field(init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_ring", deque(maxlen=self.ring_max))

    async def publish(self, event: Dict[str, Any]) -> None:
        if "id" not in event:
            event = {**event, "id": str(uuid.uuid4())}
        if "ts" not in event:
            event = {**event, "ts": _now_iso()}
        async with self._lock:
            self._ring.append(dict(event))
        dead: List[asyncio.Queue] = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(dict(event))
            except asyncio.QueueFull:
                dead.append(q)
            except Exception:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)

    def subscribe(self, maxsize: int = 256) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def recent_snapshot(self) -> List[Dict[str, Any]]:
        return list(self._ring)
