from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _compute_next_run(job: Any) -> Optional[float]:
    try:
        from croniter import croniter
    except ImportError:
        croniter = None  # type: ignore
    now = time.time()
    if job.schedule_type == "interval" and job.interval_seconds > 0:
        base = job.last_run or now
        return base + job.interval_seconds
    if job.schedule_type == "once" and job.run_at > now:
        return job.run_at
    if job.schedule_type == "cron" and croniter:
        itr = croniter(job.cron_expr, now)
        return float(itr.get_next(float))
    return None


class CronWorker:
    def __init__(
        self,
        store: Any,
        enqueue: Callable[[str, dict], Any],
        tick_seconds: int,
    ) -> None:
        self.store = store
        self.enqueue = enqueue
        self.tick_seconds = tick_seconds
        self._task: Optional[asyncio.Task] = None
        self._active = False

    async def start(self) -> None:
        self._active = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._active:
            try:
                await self._tick()
            except Exception:
                logger.exception("cron tick failed")
            await asyncio.sleep(self.tick_seconds)

    async def _tick(self) -> None:
        now = time.time()
        for job in self.store.list_jobs():
            if not job.enabled or job.paused:
                continue
            nxt = job.next_run
            if nxt is None:
                job.next_run = _compute_next_run(job)
                self.store.upsert(job)
                continue
            if now < nxt:
                continue
            await self.enqueue(job.prompt, {"cron_job_id": job.id, "name": job.name})
            job.last_run = now
            job.next_run = _compute_next_run(job)
            self.store.upsert(job)
