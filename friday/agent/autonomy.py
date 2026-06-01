from __future__ import annotations

import asyncio
import json
import logging
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
    """Durable FIFO task queue persisted under .friday/task_queue.json."""

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

    async def clear_completed(self) -> int:
        async with self._lock:
            before = len(self._tasks)
            self._tasks = [task for task in self._tasks if task.status in {"pending", "running"}]
            removed = before - len(self._tasks)
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
        self._turn_lock = asyncio.Lock()
        self._active = False
        self._worker_task: Optional[asyncio.Task] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._followed_jobs: set[str] = set()
        self._exhaustion_streak: Dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.settings.autonomy_enabled

    async def start(self) -> None:
        if not self.enabled:
            return
        self._active = True
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
        next_index = parent.continuation_index + 1
        max_cont = self.settings.autonomy_max_continuations
        if next_index > max_cont:
            return
        root_id = self._root_task_id(parent)
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

    async def _enqueue_job_followup(self, ev: Dict[str, Any]) -> None:
        job_id = str(ev.get("job_id") or "")
        if not job_id or job_id in self._followed_jobs:
            return
        self._followed_jobs.add(job_id)
        if len(self._followed_jobs) > 500:
            self._followed_jobs = set(list(self._followed_jobs)[-250:])
        task = AutonomyTask(
            id=uuid.uuid4().hex,
            message=_job_followup_message(ev),
            source="job_followup",
            metadata={"job_id": job_id, "command": ev.get("command")},
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
        await self._execute_task(task, attachments=attachments)

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
                )

                stall_count = stall_counts["count"]
                root_id = self._root_task_id(task)

                if stall_count >= settings.autonomy_agent_stall_max:
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
                    mistake_log.step_budget_exhausted = False

                if mistake_log.step_budget_exhausted:
                    self._exhaustion_streak[root_id] = self._exhaustion_streak.get(root_id, 0) + 1
                else:
                    self._exhaustion_streak[root_id] = 0

                memory_label = task.message
                if task.source == "continuation":
                    memory_label = f"[continuation {task.continuation_index}] {task.message[:500]}"
                elif task.source == "job_followup":
                    memory_label = f"[job follow-up] {task.metadata.get('command', 'shell job')}"

                self.memory.append_turn(memory_label, reply, attachments=attachment_meta or None)
                await self.bus.publish({"type": "memory_saved", "client_id": task.client_id})

                if settings.soul_enabled and settings.soul_auto_update:
                    asyncio.create_task(
                        maybe_update_soul(
                            self.soul,
                            llm=llm,
                            settings=settings,
                            bus=self.bus,
                            user_message=task.message,
                            assistant_reply=reply,
                            client_id=task.client_id,
                            mistake_log=mistake_log,
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
                    mistake_log.step_budget_exhausted
                    and settings.autonomy_auto_continue
                    and task.continuation_index < settings.autonomy_max_continuations
                    and self._exhaustion_streak.get(root_id, 0) < settings.autonomy_continuation_stall_max
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
                stall_task.cancel()
                try:
                    await stall_task
                except asyncio.CancelledError:
                    pass

    async def _worker_loop(self) -> None:
        poll = max(1, self.settings.autonomy_poll_seconds)
        while self._active:
            try:
                task = await self.queue.pop_pending()
                if task:
                    await self._execute_task(task)
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
            "queue_user_tasks": self.settings.autonomy_queue_user_tasks,
            "max_continuations": self.settings.autonomy_max_continuations,
            "watchdog_enabled": self.settings.autonomy_watchdog_enabled,
            "running_shell_jobs": len(self.sessions.running_jobs()),
            "recent_tasks": tasks,
        }
