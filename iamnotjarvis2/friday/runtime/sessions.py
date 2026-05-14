from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from friday.events.bus import EventBus


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class JobSession:
    id: str
    command: str
    cwd: str
    status: JobStatus = JobStatus.RUNNING
    return_code: Optional[int] = None
    stdout_buf: List[str] = field(default_factory=list)
    stderr_buf: List[str] = field(default_factory=list)
    proc: Optional[asyncio.subprocess.Process] = None


class SessionManager:
    """Non-PTY async subprocess jobs; streams lines to the event bus."""

    def __init__(self, bus: EventBus, default_cwd: str) -> None:
        self._bus = bus
        self._default_cwd = default_cwd
        self._jobs: Dict[str, JobSession] = {}

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": j.id,
                "command": j.command,
                "cwd": j.cwd,
                "status": j.status.value,
                "return_code": j.return_code,
                "last_output": ("".join(j.stdout_buf[-3:]) + "".join(j.stderr_buf[-3:]))[-500:],
            }
            for j in self._jobs.values()
        ]

    async def spawn(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> JobSession:
        job_id = str(uuid.uuid4())
        cwd = cwd or self._default_cwd
        session = JobSession(id=job_id, command=command, cwd=cwd)
        self._jobs[job_id] = session

        await self._bus.publish(
            {
                "type": "job_created",
                "job_id": job_id,
                "command": command,
                "cwd": cwd,
            }
        )

        try:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ.copy(),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    executable="/bin/bash",
                    env=os.environ.copy(),
                )
        except Exception as exc:
            session.status = JobStatus.ERROR
            session.return_code = -1
            await self._bus.publish({"type": "job_error", "job_id": job_id, "error": str(exc)})
            return session
        session.proc = proc

        async def pump_stream(stream: Optional[asyncio.StreamReader], kind: str) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace")
                if kind == "stdout":
                    session.stdout_buf.append(text)
                else:
                    session.stderr_buf.append(text)
                await self._bus.publish(
                    {
                        "type": "job_output",
                        "job_id": job_id,
                        "stream": kind,
                        "text": text,
                    }
                )

        async def run() -> None:
            try:
                if timeout is not None:
                    await asyncio.wait_for(
                        asyncio.gather(
                            pump_stream(proc.stdout, "stdout"),
                            pump_stream(proc.stderr, "stderr"),
                            proc.wait(),
                        ),
                        timeout=timeout,
                    )
                else:
                    await asyncio.gather(
                        pump_stream(proc.stdout, "stdout"),
                        pump_stream(proc.stderr, "stderr"),
                        proc.wait(),
                    )
                session.return_code = proc.returncode or 0
                session.status = JobStatus.DONE
                await self._bus.publish(
                    {
                        "type": "job_exit",
                        "job_id": job_id,
                        "return_code": session.return_code,
                    }
                )
            except asyncio.TimeoutError:
                session.status = JobStatus.TIMEOUT
                if proc.returncode is None:
                    proc.kill()
                await self._bus.publish(
                    {"type": "job_timeout", "job_id": job_id}
                )
            except Exception as exc:
                session.status = JobStatus.ERROR
                await self._bus.publish(
                    {"type": "job_error", "job_id": job_id, "error": str(exc)}
                )

        asyncio.create_task(run())
        return session

    async def wait_job(self, job_id: str, timeout: Optional[float] = None) -> JobSession:
        session = self._jobs.get(job_id)
        if not session or session.proc is None:
            raise KeyError(job_id)
        if timeout is not None:
            await asyncio.wait_for(session.proc.wait(), timeout=timeout)
        else:
            await session.proc.wait()
        return session

    def get_job(self, job_id: str) -> Optional[JobSession]:
        return self._jobs.get(job_id)

    async def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Run a shell command to completion; publish job_* events."""
        job_id = str(uuid.uuid4())
        cwd = cwd or self._default_cwd
        session = JobSession(id=job_id, command=command, cwd=cwd)
        self._jobs[job_id] = session
        await self._bus.publish(
            {
                "type": "job_created",
                "job_id": job_id,
                "command": command,
                "cwd": cwd,
            }
        )
        try:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=os.environ.copy(),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    executable="/bin/bash",
                    env=os.environ.copy(),
                )
        except Exception as exc:
            session.status = JobStatus.ERROR
            session.return_code = -1
            await self._bus.publish({"type": "job_error", "job_id": job_id, "error": str(exc)})
            return {
                "job_id": job_id,
                "stdout": "",
                "stderr": str(exc),
                "return_code": -1,
            }
        session.proc = proc
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        async def pump(stream: Optional[asyncio.StreamReader], kind: str, buf: List[str]) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace")
                buf.append(text)
                if kind == "stdout":
                    session.stdout_buf.append(text)
                else:
                    session.stderr_buf.append(text)
                await self._bus.publish(
                    {
                        "type": "job_output",
                        "job_id": job_id,
                        "stream": kind,
                        "text": text,
                    }
                )

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    pump(proc.stdout, "stdout", stdout_chunks),
                    pump(proc.stderr, "stderr", stderr_chunks),
                    proc.wait(),
                ),
                timeout=timeout,
            )
            rc = proc.returncode or 0
            session.return_code = rc
            session.status = JobStatus.DONE if rc == 0 else JobStatus.ERROR
            await self._bus.publish(
                {"type": "job_exit", "job_id": job_id, "return_code": rc}
            )
            return {
                "job_id": job_id,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks),
                "return_code": rc,
            }
        except asyncio.TimeoutError:
            session.status = JobStatus.TIMEOUT
            session.return_code = -1
            proc.kill()
            await self._bus.publish({"type": "job_timeout", "job_id": job_id})
            return {
                "job_id": job_id,
                "stdout": "".join(stdout_chunks),
                "stderr": "".join(stderr_chunks) + "\n[timeout]",
                "return_code": -1,
            }

    def combined_output(self, job_id: str) -> str:
        j = self._jobs.get(job_id)
        if not j:
            return ""
        return "".join(j.stdout_buf) + "".join(j.stderr_buf)
