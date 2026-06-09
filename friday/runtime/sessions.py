from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from friday.config import Settings
from friday.events.bus import EventBus
from friday.runtime.shell_analysis import analyze_shell_streams
from friday.runtime.user_shell import build_host_launch

MAX_STORED_JOBS = 48


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    TIMEOUT = "timeout"
    STOPPED = "stopped"


@dataclass
class JobSession:
    id: str
    command: str
    cwd: str
    status: JobStatus = JobStatus.RUNNING
    return_code: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    last_output_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    stdout_buf: List[str] = field(default_factory=list)
    stderr_buf: List[str] = field(default_factory=list)
    proc: Optional[asyncio.subprocess.Process] = None
    stop_requested: bool = False
    kill_process_group: bool = False
    background: bool = False
    autonomy_turn_source: Optional[str] = None


class SessionManager:
    """Non-PTY async subprocess jobs; streams lines to the event bus."""

    def __init__(self, bus: EventBus, default_cwd: str, settings: Settings) -> None:
        self._bus = bus
        self._default_cwd = default_cwd
        self._settings = settings
        self._jobs: Dict[str, JobSession] = {}

    def _stdout_text(self, session: JobSession) -> str:
        return "".join(session.stdout_buf)

    def _stderr_text(self, session: JobSession) -> str:
        return "".join(session.stderr_buf)

    def job_dict(self, session: JobSession, preview_chars: int = 8000) -> Dict[str, Any]:
        out = self._stdout_text(session)
        err = self._stderr_text(session)
        preview = ((out[-200:] if out else "") + (err[-200:] if err else ""))[-500:]
        rc = session.return_code
        payload: Dict[str, Any] = {
            "id": session.id,
            "command": session.command,
            "cwd": session.cwd,
            "status": session.status.value,
            "return_code": rc,
            "created_at": session.created_at,
            "finished_at": session.finished_at,
            "runtime_seconds": (
                int((session.finished_at or time.time()) - session.created_at)
                if session.created_at
                else None
            ),
            "seconds_since_output": int(time.time() - session.last_output_at),
            "last_output": preview,
            "stdout_preview": out[-preview_chars:] if out else "",
            "stderr_preview": err[-preview_chars:] if err else "",
        }
        if session.status != JobStatus.RUNNING:
            payload["outcome"] = analyze_shell_streams(out, err, rc)
        return payload

    async def _publish_job_finished(self, session: JobSession) -> None:
        out = self._stdout_text(session)
        err = self._stderr_text(session)
        rc = session.return_code
        status = session.status.value
        short = session.id[:8]
        outcome = analyze_shell_streams(out, err, rc)
        if session.status == JobStatus.DONE:
            if outcome.get("suspect_failure") and rc == 0:
                report = (
                    f"Shell job {short}… exit 0 but output suggests failure ({outcome['summary']})"
                )
            else:
                report = f"Shell job {short}… finished OK (exit {rc})"
        elif session.status == JobStatus.STOPPED:
            report = f"Shell job {short}… stopped by request (exit {rc})"
        elif session.status == JobStatus.TIMEOUT:
            report = f"Shell job {short}… timed out"
        elif session.status == JobStatus.ERROR:
            hint = outcome.get("summary", "")
            report = (
                f"Shell job {short}… failed (exit {rc})"
                + (f" — {hint}" if hint else "")
            )
        else:
            report = f"Shell job {short}… ended ({status}, exit {rc})"
        await self._bus.publish(
            {
                "type": "job_finished",
                "job_id": session.id,
                "command": session.command,
                "cwd": session.cwd,
                "status": status,
                "return_code": rc,
                "report": report,
                "outcome": outcome,
                "stdout_tail": out[-6000:],
                "stderr_tail": err[-6000:],
                "background": session.background,
                "autonomy_turn_source": session.autonomy_turn_source,
            }
        )

    def _prune_completed(self) -> None:
        if len(self._jobs) <= MAX_STORED_JOBS:
            return
        done = [
            (jid, j)
            for jid, j in self._jobs.items()
            if j.status != JobStatus.RUNNING and j.finished_at is not None
        ]
        done.sort(key=lambda x: x[1].finished_at or 0.0)
        while len(self._jobs) > MAX_STORED_JOBS and done:
            jid, _ = done.pop(0)
            self._jobs.pop(jid, None)

    def clear_jobs(self, include_running: bool = False) -> int:
        """Clear completed jobs from memory. Returns count removed."""
        removed = 0
        for jid, j in list(self._jobs.items()):
            if include_running or j.status != JobStatus.RUNNING:
                if j.status != JobStatus.RUNNING:
                    self._jobs.pop(jid, None)
                    removed += 1
        return removed

    def list_jobs(self) -> List[Dict[str, Any]]:
        items = [self.job_dict(j) for j in self._jobs.values()]
        items.sort(key=lambda d: d.get("created_at") or 0.0, reverse=True)
        return items

    def running_jobs(self) -> List[JobSession]:
        return [j for j in self._jobs.values() if j.status == JobStatus.RUNNING]

    def _touch_output(self, session: JobSession) -> None:
        session.last_output_at = time.time()

    def get_job_dict(self, job_id: str) -> Optional[Dict[str, Any]]:
        session = self._jobs.get(job_id)
        if not session:
            return None
        return self.job_dict(session, preview_chars=50_000)

    def _terminate_proc_soft(self, session: JobSession, proc: Optional[asyncio.subprocess.Process]) -> None:
        if not proc or proc.returncode is not None:
            return
        try:
            if session.kill_process_group and sys.platform != "win32" and proc.pid:
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError, ChildProcessError, OSError):
            pass

    def _terminate_proc_hard(self, session: JobSession, proc: Optional[asyncio.subprocess.Process]) -> None:
        if not proc or proc.returncode is not None:
            return
        try:
            if session.kill_process_group and sys.platform != "win32" and proc.pid:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, ChildProcessError, OSError):
            pass

    async def terminate_job(self, job_id: str) -> Dict[str, Any]:
        session = self._jobs.get(job_id)
        if not session:
            return {"ok": False, "error": "job_not_found"}
        if session.status != JobStatus.RUNNING:
            return {"ok": False, "error": "job_not_running", "status": session.status.value}
        session.stop_requested = True
        proc = session.proc
        self._terminate_proc_soft(session, proc)
        await self._bus.publish(
            {"type": "job_stop_requested", "job_id": job_id, "command": session.command}
        )
        if proc and proc.returncode is None:
            asyncio.create_task(self._delayed_hard_kill(session, proc, 14.0))
        return {"ok": True, "job_id": job_id}

    async def _delayed_hard_kill(
        self, session: JobSession, proc: asyncio.subprocess.Process, delay: float
    ) -> None:
        await asyncio.sleep(delay)
        if session.status != JobStatus.RUNNING:
            return
        if proc.returncode is not None:
            return
        self._terminate_proc_hard(session, proc)

    async def _open_host_process(
        self,
        command: str,
        cwd: str,
        *,
        kill_process_tree: bool,
    ) -> Tuple[asyncio.subprocess.Process, bool]:
        launch = build_host_launch(command, self._settings)
        kw: Dict[str, Any] = dict(
            cwd=cwd,
            env=os.environ.copy(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        use_killpg = (
            kill_process_tree
            and sys.platform != "win32"
            and self._settings.unix_kill_background_group
        )
        if launch.exec_argv:
            if use_killpg:
                kw["start_new_session"] = True
            proc = await asyncio.create_subprocess_exec(*launch.exec_argv, **kw)
            return proc, use_killpg
        if launch.use_windows_cmd_shell and sys.platform == "win32":
            proc = await asyncio.create_subprocess_shell(command, **kw)
            return proc, False
        raise RuntimeError("unsupported host shell configuration")

    async def spawn(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        *,
        autonomy_turn_source: Optional[str] = None,
    ) -> JobSession:
        if timeout is not None and timeout <= 0:
            timeout = None
        job_id = str(uuid.uuid4())
        cwd = cwd or self._default_cwd
        session = JobSession(
            id=job_id,
            command=command,
            cwd=cwd,
            background=True,
            autonomy_turn_source=autonomy_turn_source,
        )
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
            proc, killpg = await self._open_host_process(command, cwd, kill_process_tree=True)
            session.kill_process_group = killpg
        except Exception as exc:
            session.status = JobStatus.ERROR
            session.return_code = -1
            session.finished_at = time.time()
            await self._bus.publish({"type": "job_error", "job_id": job_id, "error": str(exc)})
            await self._publish_job_finished(session)
            self._prune_completed()
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
                self._touch_output(session)
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
                rc = 0 if proc.returncode is None else proc.returncode
                session.return_code = rc
                if session.stop_requested:
                    session.status = JobStatus.STOPPED
                elif rc == 0:
                    session.status = JobStatus.DONE
                else:
                    session.status = JobStatus.ERROR
                await self._bus.publish(
                    {
                        "type": "job_exit",
                        "job_id": job_id,
                        "return_code": rc,
                    }
                )
            except asyncio.TimeoutError:
                session.status = JobStatus.TIMEOUT
                session.return_code = -1
                if proc.returncode is None:
                    self._terminate_proc_hard(session, proc)
                await self._bus.publish(
                    {"type": "job_timeout", "job_id": job_id}
                )
            except Exception as exc:
                session.status = JobStatus.ERROR
                session.return_code = -1
                await self._bus.publish(
                    {"type": "job_error", "job_id": job_id, "error": str(exc)}
                )
            finally:
                if session.finished_at is None:
                    session.finished_at = time.time()
                await self._publish_job_finished(session)
                self._prune_completed()

        asyncio.create_task(run())
        return session

    def get_job(self, job_id: str) -> Optional[JobSession]:
        return self._jobs.get(job_id)

    async def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: float = 300.0,
        *,
        autonomy_turn_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a shell command to completion; publish job_* events."""
        job_id = str(uuid.uuid4())
        cwd = cwd or self._default_cwd
        session = JobSession(
            id=job_id,
            command=command,
            cwd=cwd,
            background=False,
            autonomy_turn_source=autonomy_turn_source,
        )
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
            proc, killpg = await self._open_host_process(command, cwd, kill_process_tree=False)
            session.kill_process_group = killpg
        except Exception as exc:
            session.status = JobStatus.ERROR
            session.return_code = -1
            session.finished_at = time.time()
            await self._bus.publish({"type": "job_error", "job_id": job_id, "error": str(exc)})
            await self._publish_job_finished(session)
            self._prune_completed()
            stdout_empty, stderr_exc = "", str(exc)
            return {
                "job_id": job_id,
                "stdout": stdout_empty,
                "stderr": stderr_exc,
                "return_code": -1,
                "outcome": analyze_shell_streams(stdout_empty, stderr_exc, -1),
            }
        session.proc = proc
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []

        async def pump(stream: Optional[asyncio.StreamReader], kind: str, buf: List[str]) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(16384)
                if not chunk:
                    break
                text = chunk.decode(errors="replace")
                buf.append(text)
                if kind == "stdout":
                    session.stdout_buf.append(text)
                else:
                    session.stderr_buf.append(text)
                self._touch_output(session)
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
            rc = 0 if proc.returncode is None else proc.returncode
            session.return_code = rc
            if session.stop_requested:
                session.status = JobStatus.STOPPED
            elif rc == 0:
                session.status = JobStatus.DONE
            else:
                session.status = JobStatus.ERROR
            session.finished_at = time.time()
            await self._bus.publish(
                {"type": "job_exit", "job_id": job_id, "return_code": rc}
            )
            await self._publish_job_finished(session)
            self._prune_completed()
            sout = "".join(stdout_chunks)
            serr = "".join(stderr_chunks)
            return {
                "job_id": job_id,
                "stdout": sout,
                "stderr": serr,
                "return_code": rc,
                "outcome": analyze_shell_streams(sout, serr, rc),
            }
        except asyncio.TimeoutError:
            session.status = JobStatus.TIMEOUT
            session.return_code = -1
            session.finished_at = time.time()
            self._terminate_proc_hard(session, proc)
            await self._bus.publish({"type": "job_timeout", "job_id": job_id})
            await self._publish_job_finished(session)
            self._prune_completed()
            sout = "".join(stdout_chunks)
            serr = "".join(stderr_chunks) + "\n[timeout]"
            return {
                "job_id": job_id,
                "stdout": sout,
                "stderr": serr,
                "return_code": -1,
                "outcome": analyze_shell_streams(sout, serr, -1),
            }
        except Exception as exc:
            session.status = JobStatus.ERROR
            session.return_code = -1
            session.finished_at = time.time()
            await self._bus.publish({"type": "job_error", "job_id": job_id, "error": str(exc)})
            await self._publish_job_finished(session)
            self._prune_completed()
            sout = "".join(stdout_chunks)
            serr = "".join(stderr_chunks) + f"\n{exc}"
            return {
                "job_id": job_id,
                "stdout": sout,
                "stderr": serr,
                "return_code": -1,
                "outcome": analyze_shell_streams(sout, serr, -1),
            }
