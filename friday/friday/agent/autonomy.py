from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from friday.agent.checkpoints import CheckpointManager
from friday.agent.delegation import run_delegate_tasks
from friday.agent.memory import MemoryStore
from friday.agent.orchestrator import run_orchestrated_turn
from friday.agent.persistent_memory import PersistentMemoryStore, build_combined_memory_context
from friday.agent.refs import expand_references
from friday.agent.soul import SoulStore
from friday.agent.turn_context import AgentExtras, build_agent_extras
from friday.paths import AUTONOMY_STATE_FILE
from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.llm.soul_update import maybe_update_soul
from friday.llm.usage import TokenUsageStore
from friday.runtime.files import FileRegistry
from friday.runtime.persist import atomic_write_json
from friday.runtime.sessions import SessionManager
from friday.runtime.watchdog import JobWatchdog

logger = logging.getLogger(__name__)

TASK_SOURCES = frozenset({"user", "continuation", "job_followup", "manual", "cron"})

# Shell jobs started during these turns must not enqueue another job_followup.
_JOB_FOLLOWUP_SKIP_TURN_SOURCES = frozenset({"job_followup", "delegate", "continuation"})

_AUTONOMOUS_SOURCES = frozenset({"job_followup", "continuation", "cron"})

_DIAGNOSTIC_PATTERNS = (
    re.compile(r"--version\b", re.I),
    re.compile(r"\btest-path\b", re.I),
    re.compile(r"\bwhere\.exe\b", re.I),
    re.compile(r"\bwhere\s+\S", re.I),
    re.compile(r"\bget-command\b", re.I),
    re.compile(r"\bpip\s+(show|list|install)\b", re.I),
    re.compile(r"\bwinget\s+install\b", re.I),
    re.compile(r"\btesseract\b", re.I),
    re.compile(r"\bpytesseract\b", re.I),
    re.compile(r"\beasyocr\b", re.I),
    re.compile(r"\bchoco\s+install\b", re.I),
    re.compile(r"\bcommand\s+-v\b", re.I),
)

_RECENT_COMMAND_WINDOW = 30


def normalize_shell_command(command: str) -> str:
    return " ".join(str(command or "").lower().split())


def is_diagnostic_shell_command(command: str) -> bool:
    """True when command is a path/version/OCR probe, not general file work."""
    cmd = str(command or "").strip()
    if not cmd:
        return False
    lower = cmd.lower()
    for pattern in _DIAGNOSTIC_PATTERNS:
        if pattern.search(lower):
            return True
    if "get-childitem" in lower:
        if re.search(r"tesseract\.exe|programs\\tesseract|tesseract-ocr", lower):
            return True
        if re.search(r"-filter\s+[\w.-]+\.exe", lower):
            return True
        if re.search(r"['\"][^'\"]*\.exe['\"]", lower):
            return True
    return False


def _stderr_has_real_errors(stderr: str) -> bool:
    text = stderr.strip()
    if not text:
        return False
    if "CLIXML" in text and ' S="Error"' not in text:
        return False
    return True


def _job_outcome_is_clean_success(ev: Dict[str, Any]) -> bool:
    outcome = ev.get("outcome") or {}
    if outcome.get("suspect_failure"):
        return False
    status = str(ev.get("status") or "").lower()
    if status in {"stopped", "timeout", "error"}:
        return False
    report = str(ev.get("report") or "").lower()
    if any(token in report for token in ("watchdog", "stall loop", "no output for")):
        return False
    return ev.get("return_code") == 0 and not _stderr_has_real_errors(str(ev.get("stderr_tail") or ""))


def _load_autonomy_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_autonomy_state(path: Path, state: Dict[str, Any]) -> None:
    try:
        atomic_write_json(path, state)
    except OSError as exc:
        logger.warning("autonomy state persist failed: %s", exc)


@dataclass
class AutonomyTask:
    id: str
    message: str
    source: str = "user"
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    client_id: Optional[str] = None
    attachments: Optional[List[str]] = None
    continuation_index: int = 0
    parent_task_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutonomyTask":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            message=str(data.get("message") or ""),
            source=str(data.get("source") or "user"),
            status=str(data.get("status") or "pending"),
            created_at=float(data.get("created_at") or time.time()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            client_id=data.get("client_id"),
            attachments=list(data.get("attachments") or []) or None,
            continuation_index=int(data.get("continuation_index") or 0),
            parent_task_id=data.get("parent_task_id"),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
        )


class TaskQueue:
    """Durable FIFO task queue persisted under FRIDAY_DIR/task_queue.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tasks: List[AutonomyTask] = []
        self._lock = asyncio.Lock()
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        items = data.get("tasks")
        if not isinstance(items, list):
            return
        self._tasks = [AutonomyTask.from_dict(item) for item in items if isinstance(item, dict)]

    def _save(self) -> None:
        try:
            atomic_write_json(self.path, {"tasks": [task.to_dict() for task in self._tasks[-200:]]})
        except OSError as exc:
            logger.warning("task queue persist failed: %s", exc)

    async def enqueue(self, task: AutonomyTask) -> AutonomyTask:
        async with self._lock:
            self._tasks.append(task)
            self._save()
        return task

    async def list_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        async with self._lock:
            items = list(self._tasks[-limit:])
        return [task.to_dict() for task in reversed(items)]

    async def pending_count(self) -> int:
        async with self._lock:
            return sum(1 for task in self._tasks if task.status == "pending")

    async def pop_pending(self) -> Optional[AutonomyTask]:
        async with self._lock:
            for task in self._tasks:
                if task.status == "pending":
                    task.status = "running"
                    task.started_at = time.time()
                    self._save()
                    return task
        return None

    async def mark_done(self, task_id: str) -> None:
        async with self._lock:
            for task in self._tasks:
                if task.id == task_id:
                    task.status = "done"
                    task.finished_at = time.time()
                    self._save()
                    return

    async def mark_failed(self, task_id: str, error: str) -> None:
        async with self._lock:
            for task in self._tasks:
                if task.id == task_id:
                    task.status = "failed"
                    task.finished_at = time.time()
                    task.error = error[:500]
                    self._save()
                    return

    async def update_metadata(self, task_id: str, metadata: Dict[str, Any]) -> None:
        async with self._lock:
            for task in self._tasks:
                if task.id == task_id:
                    task.metadata = dict(metadata)
                    self._save()
                    return

    async def clear_completed(self) -> int:
        async with self._lock:
            before = len(self._tasks)
            self._tasks = [task for task in self._tasks if task.status in {"pending", "running"}]
            removed = before - len(self._tasks)
            if removed:
                self._save()
            return removed

    async def clear_all(self) -> int:
        async with self._lock:
            removed = len(self._tasks)
            self._tasks = []
            if removed:
                self._save()
            return removed

    async def cancel_pending_for_root(self, root_id: str, *, reason: str) -> int:
        async with self._lock:
            cancelled = 0
            for task in self._tasks:
                if task.status != "pending":
                    continue
                task_root = str(task.metadata.get("root_task_id") or task.parent_task_id or task.id)
                if task_root != root_id and task.parent_task_id != root_id:
                    continue
                task.status = "cancelled"
                task.finished_at = time.time()
                task.error = reason[:500]
                cancelled += 1
            if cancelled:
                self._save()
            return cancelled

    async def active_count_by_source(self, source: str) -> int:
        async with self._lock:
            return sum(
                1
                for task in self._tasks
                if task.source == source and task.status in {"pending", "running"}
            )

    async def trim_source_backlog(
        self,
        source: str,
        *,
        keep_pending: int = 0,
        reason: str,
    ) -> int:
        """Cancel excess pending/running tasks for a source, keeping newest pending up to keep_pending."""
        async with self._lock:
            active = [
                task
                for task in self._tasks
                if task.source == source and task.status in {"pending", "running"}
            ]
            if not active:
                return 0
            pending = sorted(
                [task for task in active if task.status == "pending"],
                key=lambda task: task.created_at,
                reverse=True,
            )
            keep_ids = {task.id for task in pending[: max(0, keep_pending)]}
            cancelled = 0
            for task in active:
                if task.status == "pending" and task.id in keep_ids:
                    continue
                task.status = "cancelled"
                task.finished_at = time.time()
                task.error = reason[:500]
                cancelled += 1
            if cancelled:
                self._save()
            return cancelled


def _continuation_message(index: int, max_cont: int, last_reply: str) -> str:
    preview = (last_reply or "").strip()[:2000]
    return (
        f"[Autonomous continuation {index + 1}/{max_cont}] "
        "The previous turn hit the step budget before the task was complete.\n\n"
        f"Last progress:\n{preview or '(no assistant text)'}\n\n"
        "Continue from where you left off. Do not repeat completed work. "
        "Use tools to finish remaining steps, then give a brief summary."
    )


def _job_followup_message(ev: Dict[str, Any]) -> str:
    command = ev.get("command") or "unknown"
    status = ev.get("status") or "unknown"
    rc = ev.get("return_code")
    report = ev.get("report") or ""
    stderr = str(ev.get("stderr_tail") or "")[-1500:]
    stdout = str(ev.get("stdout_tail") or "")[-1500:]
    return (
        "[Autonomous job follow-up] A background shell job finished.\n\n"
        f"Command: {command}\n"
        f"Status: {status}\n"
        f"Exit code: {rc}\n"
        f"Report: {report}\n\n"
        f"stdout tail:\n{stdout or '(empty)'}\n\n"
        f"stderr tail:\n{stderr or '(empty)'}\n\n"
        "Review the outcome. If follow-up is needed (fix errors, deliver results, "
        "restart services, notify the user), act now with tools. "
        "If nothing is required, reply in one sentence that no action was needed."
    )


class AutonomyEngine:
    """Background worker: durable queue, auto-continuation, shell job follow-ups."""

    def __init__(
        self,
        *,
        settings: Settings,
        bus: EventBus,
        sessions: SessionManager,
        memory: MemoryStore,
        soul: SoulStore,
        usage: TokenUsageStore,
        registry: FileRegistry,
        queue_path: Path,
        watchdog: Optional[JobWatchdog] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        persistent_memory: Optional[PersistentMemoryStore] = None,
        hook_runner: Any = None,
        code_exec_runner: Any = None,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.sessions = sessions
        self.memory = memory
        self.soul = soul
        self.usage = usage
        self.registry = registry
        self.watchdog = watchdog
        self.checkpoint_manager = checkpoint_manager
        self.persistent_memory = persistent_memory
        self.hook_runner = hook_runner
        self.code_exec_runner = code_exec_runner
        self.queue = TaskQueue(queue_path)
        self._state_path = AUTONOMY_STATE_FILE
        self._turn_lock = asyncio.Lock()
        self._active = False
        self._worker_task: Optional[asyncio.Task] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running_executions: set[asyncio.Task] = set()
        self._followed_jobs: set[str] = set()
        self._exhaustion_streak: Dict[str, int] = {}
        self._loop_broken_roots: set[str] = set()
        self._followup_count_by_root: Dict[str, int] = {}
        self._recent_commands: List[str] = []
        self._job_followup_loop_warned = False
        self._load_persisted_state()

    @property
    def enabled(self) -> bool:
        return self.settings.autonomy_enabled

    def _load_persisted_state(self) -> None:
        state = _load_autonomy_state(self._state_path)
        self._followed_jobs = set(state.get("followed_jobs") or [])
        self._recent_commands = list(state.get("recent_commands") or [])[-_RECENT_COMMAND_WINDOW:]
        self._followup_count_by_root = {
            str(k): int(v)
            for k, v in (state.get("followup_count_by_root") or {}).items()
        }

    def _persist_state(self) -> None:
        _save_autonomy_state(
            self._state_path,
            {
                "followed_jobs": list(self._followed_jobs)[-500:],
                "recent_commands": self._recent_commands[-_RECENT_COMMAND_WINDOW:],
                "followup_count_by_root": self._followup_count_by_root,
            },
        )

    def _is_loop_broken(self, root_id: str) -> bool:
        return root_id in self._loop_broken_roots

    def _mark_loop_broken(self, root_id: str, *, reason: str) -> None:
        self._loop_broken_roots.add(root_id)

    def _command_recently_seen(self, command: str) -> bool:
        normalized = normalize_shell_command(command)
        if not normalized:
            return False
        return normalized in self._recent_commands

    def _record_command(self, command: str) -> None:
        normalized = normalize_shell_command(command)
        if not normalized:
            return
        self._recent_commands.append(normalized)
        self._recent_commands = self._recent_commands[-_RECENT_COMMAND_WINDOW:]
        self._persist_state()

    async def start(self) -> None:
        if not self.enabled:
            return
        self._active = True
        trimmed = await self.queue.trim_source_backlog(
            "job_followup",
            keep_pending=0,
            reason="job_followup backlog cleared on startup",
        )
        cont_trimmed = await self.queue.trim_source_backlog(
            "continuation",
            keep_pending=0,
            reason="continuation backlog cleared on startup",
        )
        if trimmed or cont_trimmed:
            logger.warning(
                "trimmed stale autonomy tasks on startup (followup=%s continuation=%s)",
                trimmed,
                cont_trimmed,
            )
            await self.bus.publish(
                {
                    "type": "autonomy_loop_broken",
                    "reason": "autonomy_backlog_trimmed",
                    "cancelled": trimmed + cont_trimmed,
                }
            )
        cleared = await self.queue.clear_completed()
        if cleared:
            logger.info("cleared %s completed autonomy task(s) on startup", cleared)
        self._worker_task = asyncio.create_task(self._worker_loop())
        if self.settings.autonomy_job_followup:
            self._listener_task = asyncio.create_task(self._event_listener())
        pending = await self.queue.pending_count()
        await self.bus.publish(
            {
                "type": "autonomy_started",
                "pending_tasks": pending,
                "poll_seconds": self.settings.autonomy_poll_seconds,
            }
        )
        logger.info("autonomy worker started (pending=%s)", pending)

    async def stop(self) -> None:
        self._active = False
        for task in (self._worker_task, self._listener_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._worker_task = None
        self._listener_task = None

    async def enqueue_user(
        self,
        message: str,
        *,
        client_id: Optional[str] = None,
        attachments: Optional[List[str]] = None,
    ) -> AutonomyTask:
        task = AutonomyTask(
            id=uuid.uuid4().hex,
            message=message.strip(),
            source="user",
            client_id=client_id,
            attachments=attachments,
        )
        await self.queue.enqueue(task)
        await self.bus.publish(
            {
                "type": "autonomy_task_enqueued",
                "task_id": task.id,
                "source": task.source,
                "client_id": client_id,
                "preview": task.message[:200],
            }
        )
        return task

    async def enqueue_cron(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> AutonomyTask:
        task = AutonomyTask(
            id=uuid.uuid4().hex,
            message=message.strip(),
            source="cron",
            metadata=dict(metadata or {}),
        )
        await self.queue.enqueue(task)
        await self.bus.publish(
            {
                "type": "autonomy_task_enqueued",
                "task_id": task.id,
                "source": "cron",
                "preview": task.message[:200],
            }
        )
        return task

    async def enqueue_manual(self, message: str, *, client_id: Optional[str] = None) -> AutonomyTask:
        task = AutonomyTask(
            id=uuid.uuid4().hex,
            message=message.strip(),
            source="manual",
            client_id=client_id,
        )
        await self.queue.enqueue(task)
        await self.bus.publish(
            {
                "type": "autonomy_task_enqueued",
                "task_id": task.id,
                "source": task.source,
                "client_id": client_id,
                "preview": task.message[:200],
            }
        )
        return task

    @staticmethod
    def _root_task_id(task: AutonomyTask) -> str:
        return str(task.metadata.get("root_task_id") or task.id)

    async def _enqueue_continuation(
        self,
        *,
        parent: AutonomyTask,
        last_reply: str,
    ) -> None:
        if parent.source == "job_followup":
            return
        if self.settings.autonomy_continue_user_only and parent.source != "user":
            return
        next_index = parent.continuation_index + 1
        max_cont = self.settings.autonomy_max_continuations
        if next_index > max_cont:
            return
        root_id = self._root_task_id(parent)
        if self._is_loop_broken(root_id):
            return
        streak = self._exhaustion_streak.get(root_id, 0)
        if streak >= self.settings.autonomy_continuation_stall_max:
            await self.queue.cancel_pending_for_root(
                root_id,
                reason="continuation loop: step budget exhausted repeatedly",
            )
            await self.bus.publish(
                {
                    "type": "autonomy_loop_broken",
                    "root_task_id": root_id,
                    "reason": "continuation_stall_max",
                    "streak": streak,
                }
            )
            return
        task = AutonomyTask(
            id=uuid.uuid4().hex,
            message=_continuation_message(next_index, max_cont, last_reply),
            source="continuation",
            continuation_index=next_index,
            parent_task_id=parent.id,
            client_id=parent.client_id,
            metadata={"root_task_id": root_id},
        )
        await self.queue.enqueue(task)
        await self.bus.publish(
            {
                "type": "autonomy_continuation",
                "task_id": task.id,
                "parent_task_id": parent.id,
                "continuation_index": next_index,
                "client_id": parent.client_id,
            }
        )

    async def _active_job_followup_count(self) -> int:
        return await self.queue.active_count_by_source("job_followup")

    async def _enqueue_job_followup(self, ev: Dict[str, Any]) -> None:
        job_id = str(ev.get("job_id") or "")
        if not job_id or job_id in self._followed_jobs:
            return
        command = str(ev.get("command") or "")
        root_id = str(ev.get("root_task_id") or job_id)

        if self._is_loop_broken(root_id):
            self._followed_jobs.add(job_id)
            return
        if is_diagnostic_shell_command(command):
            self._followed_jobs.add(job_id)
            self._record_command(command)
            return
        if _job_outcome_is_clean_success(ev):
            self._followed_jobs.add(job_id)
            return
        if self._command_recently_seen(command):
            self._followed_jobs.add(job_id)
            return
        # run_shell already returns output in the same agent turn; only background jobs need follow-up.
        if not ev.get("background"):
            self._followed_jobs.add(job_id)
            return
        turn_source = str(ev.get("autonomy_turn_source") or "").strip()
        if turn_source in _JOB_FOLLOWUP_SKIP_TURN_SOURCES:
            self._followed_jobs.add(job_id)
            return
        max_per_root = self.settings.autonomy_job_followup_max_per_root
        if max_per_root > 0:
            root_count = self._followup_count_by_root.get(root_id, 0)
            if root_count >= max_per_root:
                self._followed_jobs.add(job_id)
                return
        max_pending = max(1, self.settings.autonomy_job_followup_max_pending)
        active = await self._active_job_followup_count()
        if active >= max_pending:
            self._followed_jobs.add(job_id)
            if not self._job_followup_loop_warned:
                self._job_followup_loop_warned = True
                await self.bus.publish(
                    {
                        "type": "autonomy_loop_broken",
                        "reason": "job_followup_max_pending",
                        "active": active,
                        "max_pending": max_pending,
                    }
                )
            return
        self._followed_jobs.add(job_id)
        if len(self._followed_jobs) > 500:
            self._followed_jobs = set(list(self._followed_jobs)[-250:])
        self._followup_count_by_root[root_id] = self._followup_count_by_root.get(root_id, 0) + 1
        self._record_command(command)
        self._persist_state()
        task = AutonomyTask(
            id=uuid.uuid4().hex,
            message=_job_followup_message(ev),
            source="job_followup",
            metadata={"job_id": job_id, "command": ev.get("command"), "root_task_id": root_id},
        )
        await self.queue.enqueue(task)
        await self.bus.publish(
            {
                "type": "autonomy_task_enqueued",
                "task_id": task.id,
                "source": task.source,
                "preview": task.message[:200],
                "job_id": job_id,
            }
        )

    async def run_turn_immediate(
        self,
        message: str,
        *,
        client_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        source: str = "user",
        continuation_index: int = 0,
        parent_task_id: Optional[str] = None,
    ) -> None:
        """Run a turn now (serialized). Used when user tasks bypass the queue."""
        task = AutonomyTask(
            id=uuid.uuid4().hex,
            message=message.strip(),
            source=source,
            status="running",
            started_at=time.time(),
            client_id=client_id,
            attachments=[str(a.get("path")) for a in attachments] if attachments else None,
            continuation_index=continuation_index,
            parent_task_id=parent_task_id,
        )
        current = asyncio.current_task()
        if current is not None:
            self._running_executions.add(current)
        try:
            await self._execute_task(task, attachments=attachments)
        finally:
            if current is not None:
                self._running_executions.discard(current)

    async def _resolve_attachments(
        self, paths: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        if not paths:
            return []
        workdir = Path(self.settings.agent_workdir).resolve()
        resolved: List[Dict[str, Any]] = []
        for rel in paths:
            candidate = (workdir / rel).resolve()
            if not candidate.is_file():
                continue
            meta = self.registry.register(candidate, workdir, name=candidate.name)
            meta["path"] = rel
            resolved.append(meta)
        return resolved

    def _start_stall_counter(self) -> Tuple[asyncio.Task, Dict[str, int]]:
        stalls: Dict[str, int] = {"count": 0}
        q = self.bus.subscribe()

        async def drain() -> None:
            try:
                while True:
                    ev = await q.get()
                    if ev.get("type") == "agent_stall_detected":
                        stalls["count"] += 1
            except asyncio.CancelledError:
                pass
            finally:
                self.bus.unsubscribe(q)

        return asyncio.create_task(drain()), stalls

    def _start_plan_tracker(self, task: AutonomyTask) -> asyncio.Task:
        q = self.bus.subscribe()

        async def persist_plan() -> None:
            await self.queue.update_metadata(task.id, task.metadata)
            await self.bus.publish(
                {
                    "type": "autonomy_task_updated",
                    "task_id": task.id,
                    "client_id": task.client_id,
                    "metadata": task.metadata,
                }
            )

        def update_step(step_id: str, status: str) -> None:
            plan = task.metadata.get("plan")
            if not isinstance(plan, dict):
                return
            steps = plan.get("steps")
            if not isinstance(steps, list):
                return
            for step in steps:
                if isinstance(step, dict) and step.get("id") == step_id:
                    step["status"] = status
                    return

        async def drain() -> None:
            try:
                while True:
                    ev = await q.get()
                    ev_client = ev.get("client_id")
                    if ev_client and task.client_id and ev_client != task.client_id:
                        continue
                    ev_type = ev.get("type")
                    if ev_type == "plan_created":
                        plan = ev.get("plan") if isinstance(ev.get("plan"), dict) else {}
                        task.metadata["plan"] = plan
                        task.metadata["plan_status"] = "running"
                        await persist_plan()
                    elif ev_type == "plan_step_started":
                        update_step(str(ev.get("step_id") or ""), "running")
                        await persist_plan()
                    elif ev_type == "plan_step_retry":
                        update_step(str(ev.get("step_id") or ""), "retrying")
                        await persist_plan()
                    elif ev_type == "plan_step_complete":
                        update_step(str(ev.get("step_id") or ""), "done")
                        await persist_plan()
                    elif ev_type == "plan_step_failed":
                        update_step(str(ev.get("step_id") or ""), "failed")
                        task.metadata["plan_status"] = "degraded"
                        await persist_plan()
                    elif ev_type == "plan_complete":
                        task.metadata["plan_status"] = "done" if not ev.get("failed") else "degraded"
                        task.metadata["plan_failed_steps"] = int(ev.get("failed") or 0)
                        await persist_plan()
            except asyncio.CancelledError:
                pass
            finally:
                self.bus.unsubscribe(q)

        return asyncio.create_task(drain())

    async def _execute_task(
        self,
        task: AutonomyTask,
        *,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        settings = self.settings
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            await self.queue.mark_failed(task.id, "Azure OpenAI is not configured.")
            await self.bus.publish(
                {
                    "type": "chat_error",
                    "error": "Azure OpenAI is not configured.",
                    "client_id": task.client_id,
                    "autonomous": task.source != "user",
                }
            )
            return

        async with self._turn_lock:
            llm = LLMClient(settings=settings, usage_store=self.usage)
            proactive = task.source in {"continuation", "job_followup", "manual", "cron"}
            await self.bus.publish(
                {
                    "type": "autonomy_task_started",
                    "task_id": task.id,
                    "source": task.source,
                    "client_id": task.client_id,
                    "proactive": proactive,
                }
            )
            stall_task, stall_counts = self._start_stall_counter()
            plan_task = self._start_plan_tracker(task)
            try:
                self.usage.start_turn()
                raw_message = task.message.strip()
                if raw_message.lower().startswith("/rollback"):
                    parts = raw_message.split(maxsplit=1)
                    cp_arg = parts[1].strip() if len(parts) > 1 else "last"
                    if self.checkpoint_manager and settings.checkpoints_enabled:
                        result = self.checkpoint_manager.rollback(cp_arg)
                        reply = (
                            f"Rolled back checkpoint {result.get('checkpoint_id')}: "
                            f"{', '.join(result.get('restored') or [])}"
                            if result.get("ok")
                            else f"Rollback failed: {result.get('error')}"
                        )
                    else:
                        reply = "Checkpoints are disabled."
                    await self.bus.publish(
                        {
                            "type": "chat_complete",
                            "reply": reply,
                            "client_id": task.client_id,
                            "outputs": [],
                            "autonomous": proactive,
                            "task_source": task.source,
                            "task_id": task.id,
                        }
                    )
                    await self.queue.mark_done(task.id)
                    return

                memory_context = self.memory.build_context()
                session_uploads = self.memory.recent_attachments()
                soul_context = ""
                if settings.soul_enabled and self.persistent_memory:
                    soul_context = build_combined_memory_context(
                        settings, self.soul, self.persistent_memory
                    )
                elif settings.soul_enabled:
                    soul_context = self.soul.build_context(settings.soul_max_context_chars)
                if soul_context:
                    await self.bus.publish({"type": "soul_loaded", "chars": len(soul_context)})

                workdir_path = Path(settings.agent_workdir).resolve()
                task_message = raw_message
                if settings.refs_enabled:
                    task_message, _ = expand_references(
                        task_message,
                        workdir_path,
                        max_file_chars=settings.refs_max_file_chars,
                        max_total_chars=settings.refs_max_total_chars,
                        allow_url_fetch=settings.refs_allow_url_fetch,
                    )
                if memory_context:
                    await self.bus.publish(
                        {"type": "memory_loaded", "turns": len(self.memory.recent_turns_list())}
                    )

                attachment_meta = attachments or []
                if not attachment_meta and task.attachments:
                    attachment_meta = await self._resolve_attachments(task.attachments)

                async def _delegate_runner(
                    tasks: List[Dict[str, Any]], share_context: bool = False
                ) -> str:
                    return await run_delegate_tasks(
                        tasks,
                        llm=llm,
                        bus=self.bus,
                        sessions=self.sessions,
                        workdir=settings.agent_workdir,
                        allow_shell=settings.allow_shell,
                        settings=settings,
                        file_registry=self.registry,
                        soul_store=self.soul,
                        client_id=task.client_id,
                        share_context=share_context,
                        parent_summary=task_message if share_context else "",
                    )

                extras = build_agent_extras(
                    settings,
                    workdir_path,
                    checkpoint_manager=self.checkpoint_manager,
                    persistent_memory=self.persistent_memory,
                    hook_runner=self.hook_runner,
                    delegate_runner=_delegate_runner,
                    code_exec_runner=self.code_exec_runner,
                )

                reply, outputs, mistake_log = await run_orchestrated_turn(
                    task_message,
                    llm=llm,
                    bus=self.bus,
                    sessions=self.sessions,
                    workdir=settings.agent_workdir,
                    allow_shell=settings.allow_shell,
                    max_steps=settings.max_agent_steps,
                    settings=settings,
                    memory_context=memory_context,
                    soul_context=soul_context,
                    attachments=attachment_meta or None,
                    session_uploads=session_uploads,
                    file_registry=self.registry,
                    soul_store=self.soul,
                    client_id=task.client_id,
                    agent_extras=extras,
                    autonomy_turn_source=task.source,
                )

                stall_count = stall_counts["count"]
                root_id = self._root_task_id(task)
                loop_broken = False

                if stall_count >= settings.autonomy_agent_stall_max:
                    loop_broken = True
                    self._mark_loop_broken(root_id, reason="agent_stall_max")
                    await self.queue.cancel_pending_for_root(
                        root_id,
                        reason=f"agent stall loop ({stall_count} repeated tool calls)",
                    )
                    await self.bus.publish(
                        {
                            "type": "autonomy_loop_broken",
                            "root_task_id": root_id,
                            "reason": "agent_stall_max",
                            "stall_count": stall_count,
                            "task_id": task.id,
                        }
                    )
                    if self.watchdog:
                        await self.watchdog.stop_all_running(
                            reason=f"agent stall loop ({stall_count} repeated tool calls)",
                        )

                if mistake_log.step_budget_exhausted and not loop_broken:
                    self._exhaustion_streak[root_id] = self._exhaustion_streak.get(root_id, 0) + 1
                elif not loop_broken:
                    self._exhaustion_streak[root_id] = max(0, self._exhaustion_streak.get(root_id, 0) - 1)

                memory_label = task.message
                if task.source == "continuation":
                    memory_label = f"[continuation {task.continuation_index}] {task.message[:500]}"
                elif task.source == "job_followup":
                    memory_label = f"[job follow-up] {task.metadata.get('command', 'shell job')}"

                if task.source in _AUTONOMOUS_SOURCES and settings.conversation_memory_skip_autonomous:
                    summary = reply.strip()[:200] or "(no reply)"
                    self.memory.append_turn(
                        memory_label,
                        summary,
                        attachments=attachment_meta or None,
                        source=task.source,
                        autonomous=True,
                    )
                else:
                    self.memory.append_turn(
                        memory_label,
                        reply,
                        attachments=attachment_meta or None,
                        source=task.source,
                    )
                await self.bus.publish({"type": "memory_saved", "client_id": task.client_id})

                skip_soul = (
                    settings.soul_auto_update_skip_autonomous
                    and task.source in _AUTONOMOUS_SOURCES
                )
                if settings.soul_enabled and settings.soul_auto_update and not skip_soul:
                    asyncio.create_task(
                        maybe_update_soul(
                            self.soul,
                            llm=llm,
                            settings=settings,
                            bus=self.bus,
                            user_message=memory_label,
                            assistant_reply=reply,
                            client_id=task.client_id,
                            mistake_log=mistake_log,
                            task_source=task.source,
                        )
                    )

                await self.bus.publish(
                    {
                        "type": "chat_complete",
                        "reply": reply,
                        "client_id": task.client_id,
                        "outputs": outputs,
                        "usage": self.usage.snapshot("last_turn") if settings.token_usage_enabled else None,
                        "autonomous": proactive,
                        "task_source": task.source,
                        "task_id": task.id,
                    }
                )

                await self.queue.mark_done(task.id)

                if (
                    not loop_broken
                    and mistake_log.step_budget_exhausted
                    and settings.autonomy_auto_continue
                    and task.source != "job_followup"
                    and task.continuation_index < settings.autonomy_max_continuations
                    and self._exhaustion_streak.get(root_id, 0) < settings.autonomy_continuation_stall_max
                    and not self._is_loop_broken(root_id)
                ):
                    await self._enqueue_continuation(parent=task, last_reply=reply)

            except Exception as exc:
                logger.exception("autonomy task failed: %s", task.id)
                await self.queue.mark_failed(task.id, str(exc))
                await self.bus.publish(
                    {
                        "type": "chat_error",
                        "error": str(exc),
                        "client_id": task.client_id,
                        "autonomous": proactive,
                        "task_id": task.id,
                    }
                )
            finally:
                plan_task.cancel()
                try:
                    await plan_task
                except asyncio.CancelledError:
                    pass
                stall_task.cancel()
                try:
                    await stall_task
                except asyncio.CancelledError:
                    pass

    async def abort_all(self) -> Dict[str, int]:
        """Hard-stop every mission: cancel running turns, kill their shell jobs,
        suppress follow-ups/continuations, and empty the queue."""
        # Prevent the still-pending follow-up/continuation logic from re-queuing work.
        for task in await self.queue.list_tasks(limit=200):
            root = str((task.get("metadata") or {}).get("root_task_id") or task.get("id") or "")
            if root:
                self._loop_broken_roots.add(root)

        # Mark running shell jobs as already-followed so job_finished events from the
        # termination below do not spawn new follow-up tasks.
        for session in self.sessions.running_jobs():
            self._followed_jobs.add(session.id)

        stopped_jobs: List[str] = []
        if self.watchdog:
            stopped_jobs = await self.watchdog.stop_all_running(reason="cleared from UI")

        running = list(self._running_executions)
        self._running_executions.clear()
        for exec_task in running:
            exec_task.cancel()
        for exec_task in running:
            try:
                await exec_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("error awaiting aborted mission task")

        removed = await self.queue.clear_all()
        await self.bus.publish(
            {
                "type": "autonomy_aborted",
                "removed": removed,
                "stopped_jobs": len(stopped_jobs),
            }
        )
        logger.info("aborted missions from UI (removed=%s stopped_jobs=%s)", removed, len(stopped_jobs))
        return {"removed": removed, "stopped_jobs": len(stopped_jobs)}

    async def _worker_loop(self) -> None:
        poll = max(1, self.settings.autonomy_poll_seconds)
        while self._active:
            try:
                task = await self.queue.pop_pending()
                if task:
                    exec_task = asyncio.create_task(self._execute_task(task))
                    self._running_executions.add(exec_task)
                    try:
                        await exec_task
                    except asyncio.CancelledError:
                        # Mission was aborted from the UI; keep the worker alive.
                        pass
                    finally:
                        self._running_executions.discard(exec_task)
                else:
                    await asyncio.sleep(poll)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("autonomy worker loop error")
                await asyncio.sleep(poll)

    async def _event_listener(self) -> None:
        q = self.bus.subscribe()
        try:
            while self._active:
                ev = await q.get()
                if ev.get("type") != "job_finished":
                    continue
                if not self.settings.autonomy_job_followup:
                    continue
                await self._enqueue_job_followup(ev)
        except asyncio.CancelledError:
            pass
        finally:
            self.bus.unsubscribe(q)

    async def status(self) -> Dict[str, Any]:
        pending = await self.queue.pending_count()
        tasks = await self.queue.list_tasks(limit=20)
        running = sum(1 for task in tasks if task.get("status") == "running")
        return {
            "enabled": self.enabled,
            "pending": pending,
            "running": running,
            "poll_seconds": self.settings.autonomy_poll_seconds,
            "auto_continue": self.settings.autonomy_auto_continue,
            "job_followup": self.settings.autonomy_job_followup,
            "job_followup_max_pending": self.settings.autonomy_job_followup_max_pending,
            "queue_user_tasks": self.settings.autonomy_queue_user_tasks,
            "max_continuations": self.settings.autonomy_max_continuations,
            "watchdog_enabled": self.settings.autonomy_watchdog_enabled,
            "running_shell_jobs": len(self.sessions.running_jobs()),
            "recent_tasks": tasks,
        }
