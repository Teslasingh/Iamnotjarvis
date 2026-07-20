from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from friday.runtime.persist import atomic_write_json


@dataclass
class CronJob:
    id: str
    name: str
    prompt: str
    schedule_type: str = "cron"
    cron_expr: str = "0 9 * * *"
    interval_seconds: int = 0
    run_at: float = 0.0
    skill_names: List[str] = field(default_factory=list)
    enabled: bool = True
    paused: bool = False
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    delivery: str = "web"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CronJob":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            name=str(data.get("name") or "job"),
            prompt=str(data.get("prompt") or ""),
            schedule_type=str(data.get("schedule_type") or "cron"),
            cron_expr=str(data.get("cron_expr") or "0 9 * * *"),
            interval_seconds=int(data.get("interval_seconds") or 0),
            run_at=float(data.get("run_at") or 0),
            skill_names=list(data.get("skill_names") or []),
            enabled=bool(data.get("enabled", True)),
            paused=bool(data.get("paused", False)),
            last_run=data.get("last_run"),
            next_run=data.get("next_run"),
            delivery=str(data.get("delivery") or "web"),
        )


class CronStore:
    def __init__(self, path: Path, max_jobs: int) -> None:
        self.path = path
        self.max_jobs = max_jobs
        self._jobs: List[CronJob] = []
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._jobs = [CronJob.from_dict(j) for j in raw if isinstance(j, dict)]
        except (json.JSONDecodeError, OSError):
            self._jobs = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, [j.to_dict() for j in self._jobs[: self.max_jobs]])

    def list_jobs(self) -> List[CronJob]:
        return list(self._jobs)

    def get(self, job_id: str) -> Optional[CronJob]:
        return next((j for j in self._jobs if j.id == job_id), None)

    def upsert(self, job: CronJob) -> CronJob:
        for i, existing in enumerate(self._jobs):
            if existing.id == job.id:
                self._jobs[i] = job
                self.save()
                return job
        self._jobs.append(job)
        self.save()
        return job

    def delete(self, job_id: str) -> bool:
        before = len(self._jobs)
        self._jobs = [j for j in self._jobs if j.id != job_id]
        if len(self._jobs) < before:
            self.save()
            return True
        return False
