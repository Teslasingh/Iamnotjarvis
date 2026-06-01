from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from friday.config import Settings
from friday.events.bus import EventBus
from friday.runtime.sessions import JobSession, JobStatus, SessionManager

logger = logging.getLogger(__name__)


def detect_repetitive_output(text: str, *, min_repeats: int = 25, min_line_len: int = 4) -> Optional[str]:
    """Return a reason string if output looks like a tight print loop."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < min_repeats:
        return None
    tail = lines[-120:]
    line, count = Counter(tail).most_common(1)[0]
    if count >= min_repeats and len(line) >= min_line_len:
        preview = line[:80] + ("…" if len(line) > 80 else "")
        return f"repetitive output ({count}×): {preview!r}"
    return None


def inspect_running_job(
    session: JobSession,
    *,
    now: float,
    max_runtime_seconds: int,
    stall_seconds: int,
    output_loop_repeats: int,
) -> Optional[str]:
    """Return intervention reason if a running job looks stuck or runaway."""
    if session.status != JobStatus.RUNNING:
        return None

    age = now - session.created_at
    if max_runtime_seconds > 0 and age >= max_runtime_seconds:
        return f"max runtime exceeded ({int(age)}s)"

    since_output = now - session.last_output_at
    if stall_seconds > 0 and since_output >= stall_seconds:
        return f"no output for {int(since_output)}s (likely hung or waiting for input)"

    combined = "".join(session.stdout_buf) + "".join(session.stderr_buf)
    loop_reason = detect_repetitive_output(
        combined,
        min_repeats=output_loop_repeats,
    )
    if loop_reason:
        return loop_reason

    return None


class JobWatchdog:
    """Monitors all shell jobs and stops runaway or stuck processes."""

    def __init__(
        self,
        *,
        sessions: SessionManager,
        bus: EventBus,
        settings: Settings,
    ) -> None:
        self.sessions = sessions
        self.bus = bus
        self.settings = settings
        self._active = False
        self._task: Optional[asyncio.Task] = None
        self._stopped_jobs: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.settings.autonomy_watchdog_enabled

    async def start(self) -> None:
        if not self.enabled:
            return
        self._active = True
        self._task = asyncio.create_task(self._loop())
        logger.info("job watchdog started (poll=%ss)", self.settings.autonomy_watchdog_poll_seconds)

    async def stop(self) -> None:
        self._active = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def inspect_once(self) -> List[Dict[str, Any]]:
        """Scan running jobs and stop any that look stuck. Returns interventions."""
        if not self.enabled:
            return []

        now = time.time()
        settings = self.settings
        interventions: List[Dict[str, Any]] = []

        for session in self.sessions.running_jobs():
            if session.id in self._stopped_jobs:
                continue
            reason = inspect_running_job(
                session,
                now=now,
                max_runtime_seconds=settings.autonomy_job_max_runtime_seconds,
                stall_seconds=settings.autonomy_job_stall_seconds,
                output_loop_repeats=settings.autonomy_job_output_loop_repeats,
            )
            if not reason:
                continue
            result = await self.sessions.terminate_job(session.id)
            if not result.get("ok"):
                continue
            self._stopped_jobs.add(session.id)
            if len(self._stopped_jobs) > 500:
                self._stopped_jobs = set(list(self._stopped_jobs)[-250:])
            payload = {
                "type": "watchdog_job_stopped",
                "job_id": session.id,
                "command": session.command,
                "reason": reason,
            }
            interventions.append(payload)
            await self.bus.publish(payload)
            logger.warning("watchdog stopped job %s: %s", session.id[:8], reason)

        return interventions

    async def stop_all_running(self, *, reason: str) -> List[str]:
        """Emergency stop for all running shell jobs."""
        stopped: List[str] = []
        for session in self.sessions.running_jobs():
            result = await self.sessions.terminate_job(session.id)
            if result.get("ok"):
                stopped.append(session.id)
                await self.bus.publish(
                    {
                        "type": "watchdog_job_stopped",
                        "job_id": session.id,
                        "command": session.command,
                        "reason": reason,
                    }
                )
        return stopped

    async def _loop(self) -> None:
        poll = max(5, self.settings.autonomy_watchdog_poll_seconds)
        while self._active:
            try:
                await self.inspect_once()
                await asyncio.sleep(poll)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("job watchdog loop error")
                await asyncio.sleep(poll)
