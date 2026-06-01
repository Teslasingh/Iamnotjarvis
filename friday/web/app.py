from __future__ import annotations

import asyncio
import hmac
import hashlib
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from friday.agent.autonomy import AutonomyEngine
from friday.agent.checkpoints import CheckpointManager
from friday.agent.code_exec import run_execute_code
from friday.agent.memory import MemoryStore
from friday.agent.persistent_memory import PersistentMemoryStore
from friday.agent.soul import SoulStore
from friday.batch.runner import BatchRunner
from friday.hooks.registry import HookRegistry
from friday.hooks.runner import HookRunner
from friday.scheduler.store import CronJob, CronStore
from friday.scheduler.worker import CronWorker
from friday.config import Settings, get_settings
from friday.events.bus import EventBus
from friday.llm.usage import TokenUsageStore
from friday.runtime.files import FileRegistry, guess_mime, is_path_under, is_preview_image, normalize_rel_path, sanitize_filename
from friday.runtime.sessions import SessionManager
from friday.runtime.watchdog import JobWatchdog

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
AUTH_COOKIE = "friday_session"
CHUNK_SIZE = 1024 * 1024


def _session_value(settings: Settings) -> str:
    digest = hmac.new(
        settings.session_secret.encode("utf-8"),
        b"friday-authenticated",
        hashlib.sha256,
    ).hexdigest()
    return digest


def _is_authenticated(request: Request, settings: Settings) -> bool:
    if not settings.friday_password:
        return True
    cookie = request.cookies.get(AUTH_COOKIE, "")
    return hmac.compare_digest(cookie, _session_value(settings))


def _is_ws_authenticated(websocket: WebSocket, settings: Settings) -> bool:
    if not settings.friday_password:
        return True
    cookie = websocket.cookies.get(AUTH_COOKIE, "")
    return hmac.compare_digest(cookie, _session_value(settings))


def _workdir_path(settings: Settings) -> Path:
    return Path(settings.agent_workdir).resolve()


def _uploads_root(settings: Settings) -> Path:
    return _workdir_path(settings) / settings.upload_dir


def _validate_attachment_path(settings: Settings, rel_path: str) -> Path:
    workdir = _workdir_path(settings)
    uploads_root = _uploads_root(settings).resolve()
    candidate = (workdir / rel_path).resolve()
    if not candidate.is_file():
        raise HTTPException(400, f"attachment not found: {rel_path}")
    if not is_path_under(uploads_root, candidate):
        raise HTTPException(400, f"invalid attachment path: {rel_path}")
    return candidate


def _login_page(error: str = "") -> str:
    error_html = f'<div class="login-error">{error}</div>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Friday Login</title>
  <link rel="stylesheet" href="/static/app.css" />
</head>
<body>
  <main class="login-screen">
    <form method="post" action="/login" class="login-card">
      <div class="dot login-dot"></div>
      <h1>Friday</h1>
      <p>Authenticate to access this machine.</p>
      {error_html}
      <input type="password" name="password" placeholder="Password" autofocus />
      <button class="btn primary" type="submit">Unlock</button>
    </form>
  </main>
</body>
</html>"""


class ChatBody(BaseModel):
    message: str
    client_id: Optional[str] = None
    attachments: Optional[List[str]] = None


class TaskBody(BaseModel):
    message: str
    client_id: Optional[str] = None


class CronBody(BaseModel):
    name: str = "job"
    prompt: str
    cron_expr: str = "0 9 * * *"
    schedule_type: str = "cron"
    skill_names: Optional[List[str]] = None


class BatchBody(BaseModel):
    prompts: List[str]


class HookBody(BaseModel):
    id: str
    type: str = "gateway"
    events: Optional[List[str]] = None
    hook: Optional[str] = None
    match: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    workdir = settings.agent_workdir
    workdir_path = Path(workdir)
    hook_registry = HookRegistry(workdir_path / ".friday" / "hooks.json")
    hook_runner = HookRunner(
        hook_registry, timeout_seconds=settings.hooks_webhook_timeout_seconds
    )
    bus = EventBus(
        ring_max=settings.event_ring_max,
        hook_runner=hook_runner,
        hooks_enabled=settings.hooks_enabled,
    )
    workdir_path.mkdir(parents=True, exist_ok=True)
    (_workdir_path(settings) / settings.upload_dir).mkdir(parents=True, exist_ok=True)
    (_workdir_path(settings) / settings.output_dir).mkdir(parents=True, exist_ok=True)
    sessions = SessionManager(bus=bus, default_cwd=workdir, settings=settings)
    friday_dir = workdir_path / ".friday"
    memory = MemoryStore(
        recent_turns=settings.conversation_memory_max_turns,
        persist_path=friday_dir / "conversation.json",
        max_context_chars=settings.conversation_memory_max_context_chars,
        persist_enabled=settings.conversation_memory_enabled,
    )
    if not settings.conversation_memory_enabled:
        memory.clear()
    soul = SoulStore(workdir_path / "soul.md")
    soul.load()
    persistent_memory = PersistentMemoryStore(workdir_path)
    checkpoint_manager = CheckpointManager(
        friday_dir / "checkpoints",
        workdir_path,
        settings.checkpoints_max_count,
        settings.checkpoints_max_file_bytes,
    )
    usage = TokenUsageStore(
        call_log_max=settings.token_usage_call_log_max,
        persist_path=friday_dir / "token_usage.json",
        persist_enabled=settings.token_usage_enabled and settings.token_usage_persist,
    )
    app.state.settings = settings
    app.state.bus = bus
    app.state.sessions = sessions
    app.state.memory = memory
    app.state.soul = soul
    app.state.persistent_memory = persistent_memory
    app.state.checkpoint_manager = checkpoint_manager
    app.state.hook_registry = hook_registry
    app.state.usage = usage
    registry_path = workdir_path / ".friday" / "file_registry.json"
    registry = FileRegistry(registry_path, workdir_path)
    registry.register_existing_paths(workdir_path, [settings.upload_dir, settings.output_dir])
    app.state.file_registry = registry
    watchdog = JobWatchdog(sessions=sessions, bus=bus, settings=settings)
    app.state.watchdog = watchdog
    async def _code_exec_runner(code: str) -> str:
        return await run_execute_code(code, settings=settings, workdir=workdir_path)

    autonomy = AutonomyEngine(
        settings=settings,
        bus=bus,
        sessions=sessions,
        memory=memory,
        soul=soul,
        usage=usage,
        registry=registry,
        queue_path=friday_dir / "task_queue.json",
        watchdog=watchdog,
        checkpoint_manager=checkpoint_manager,
        persistent_memory=persistent_memory,
        hook_runner=hook_runner,
        code_exec_runner=_code_exec_runner,
    )
    app.state.autonomy = autonomy

    cron_store = CronStore(friday_dir / "cron_jobs.json", settings.cron_max_jobs)
    app.state.cron_store = cron_store

    async def _cron_enqueue(prompt: str, meta: dict) -> None:
        await autonomy.enqueue_cron(prompt, metadata=meta)

    cron_worker = CronWorker(cron_store, _cron_enqueue, settings.cron_tick_seconds)
    app.state.cron_worker = cron_worker

    async def _batch_run_turn(prompt: str):
        from friday.agent.orchestrator import run_orchestrated_turn
        from friday.agent.turn_context import build_agent_extras
        from friday.llm.client import LLMClient

        llm = LLMClient(settings=settings, usage_store=usage)
        extras = build_agent_extras(
            settings,
            workdir_path,
            checkpoint_manager=checkpoint_manager,
            persistent_memory=persistent_memory,
            hook_runner=hook_runner,
            delegate_runner=None,
            code_exec_runner=_code_exec_runner,
        )
        reply, outputs, mistakes = await run_orchestrated_turn(
            prompt,
            llm=llm,
            bus=bus,
            sessions=sessions,
            workdir=workdir,
            allow_shell=settings.allow_shell,
            max_steps=settings.max_agent_steps,
            settings=settings,
            soul_store=soul,
            agent_extras=extras,
        )
        return reply, outputs, mistakes

    batch_runner = BatchRunner(
        friday_dir / "batches",
        _batch_run_turn,
        settings.batch_max_parallel,
        settings.batch_max_items,
    )
    app.state.batch_runner = batch_runner

    await watchdog.start()
    await autonomy.start()
    if settings.cron_enabled:
        await cron_worker.start()
    try:
        yield
    finally:
        if settings.cron_enabled:
            await cron_worker.stop()
        await autonomy.stop()
        await watchdog.stop()
        if not settings.conversation_memory_enabled:
            memory.clear()


def create_app() -> FastAPI:
    app = FastAPI(title="Friday Agent", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index(request: Request) -> FileResponse:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            return RedirectResponse("/login", status_code=303)
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return JSONResponse({"error": "UI not built; missing static/index.html"}, status_code=500)
        return FileResponse(index_path)

    @app.get("/login")
    async def login_get(request: Request) -> HTMLResponse:
        settings: Settings = request.app.state.settings
        if _is_authenticated(request, settings):
            return RedirectResponse("/", status_code=303)
        return HTMLResponse(_login_page())

    @app.post("/login")
    async def login_post(request: Request) -> Response:
        settings: Settings = request.app.state.settings
        body = (await request.body()).decode("utf-8", errors="replace")
        form = parse_qs(body)
        password = (form.get("password") or [""])[0]
        if not settings.friday_password or hmac.compare_digest(password, settings.friday_password):
            resp = RedirectResponse("/", status_code=303)
            resp.set_cookie(
                AUTH_COOKIE,
                _session_value(settings),
                httponly=True,
                samesite="lax",
            )
            return resp
        return HTMLResponse(_login_page("Wrong password"), status_code=401)

    @app.post("/logout")
    async def logout(request: Request) -> Response:
        settings: Settings = request.app.state.settings
        memory: MemoryStore = request.app.state.memory
        if settings.conversation_memory_clear_on_logout:
            memory.clear()
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(AUTH_COOKIE)
        return resp

    @app.get("/favicon.ico")
    async def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket, replay: bool = False) -> None:
        settings: Settings = websocket.app.state.settings
        if not _is_ws_authenticated(websocket, settings):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        bus: EventBus = websocket.app.state.bus
        q = bus.subscribe()
        try:
            if replay:
                for ev in bus.recent_snapshot():
                    await websocket.send_json(ev)
            while True:
                ev = await q.get()
                await websocket.send_json(ev)
        except WebSocketDisconnect:
            bus.unsubscribe(q)

    @app.post("/api/upload")
    async def api_upload(request: Request, files: List[UploadFile] = File(...)) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        if not files:
            raise HTTPException(400, "no files provided")

        registry: FileRegistry = request.app.state.file_registry
        workdir = _workdir_path(settings)
        uploads_root = _uploads_root(settings)
        batch_id = uuid.uuid4().hex
        batch_dir = uploads_root / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Dict[str, Any]] = []
        for upload in files:
            safe_name = sanitize_filename(upload.filename or "upload")
            dest = (batch_dir / safe_name).resolve()
            if not is_path_under(workdir, dest):
                raise HTTPException(400, "invalid upload path")

            with dest.open("wb") as out:
                while True:
                    chunk = await upload.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)

            meta = registry.register(dest, workdir, name=safe_name)
            saved.append(meta)
            await request.app.state.bus.publish({"type": "file_uploaded", "file": meta})

        return {"files": saved}

    @app.get("/api/files/{file_id}")
    async def api_file_download(file_id: str, request: Request, inline: bool = False) -> FileResponse:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")

        registry: FileRegistry = request.app.state.file_registry
        workdir = _workdir_path(settings)
        path = registry.resolve_download(file_id, workdir)
        if not path:
            raise HTTPException(404, "file not found")

        meta = registry.get(file_id) or {}
        mime = meta.get("mime") or guess_mime(path)
        name = meta.get("name") or path.name
        if mime.startswith("text/") or path.suffix.lower() == ".txt":
            mime = "text/plain; charset=utf-8"
        disposition = "inline" if inline or is_preview_image(mime, name) else "attachment"
        return FileResponse(
            path,
            media_type=mime,
            filename=name,
            content_disposition_type=disposition,
        )

    @app.get("/api/files/by-path")
    async def api_file_by_path(path: str, request: Request, inline: bool = False) -> FileResponse:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")

        registry: FileRegistry = request.app.state.file_registry
        workdir = _workdir_path(settings)
        rel = normalize_rel_path(path)
        candidate = (workdir / rel).resolve()
        if not candidate.is_file() or not is_path_under(workdir, candidate):
            raise HTTPException(404, "file not found")

        meta = registry.register(candidate, workdir, name=candidate.name)
        file_id = meta["id"]
        return await api_file_download(file_id, request, inline=inline)

    @app.post("/api/chat")
    async def api_chat(body: ChatBody, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        if not body.message.strip():
            raise HTTPException(400, "empty message")
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            raise HTTPException(503, "Azure OpenAI is not configured.")
        registry: FileRegistry = request.app.state.file_registry
        autonomy: AutonomyEngine = request.app.state.autonomy
        user_message = body.message.strip()
        client_id = body.client_id

        attachment_meta: List[Dict[str, Any]] = []
        attachment_paths: List[str] = []
        for rel_path in body.attachments or []:
            path = _validate_attachment_path(settings, rel_path)
            meta = registry.register(path, _workdir_path(settings), name=path.name)
            meta["path"] = normalize_rel_path(meta.get("path") or rel_path)
            attachment_meta.append(meta)
            attachment_paths.append(meta["path"])

        if settings.autonomy_enabled and settings.autonomy_queue_user_tasks:
            task = await autonomy.enqueue_user(
                user_message,
                client_id=client_id,
                attachments=attachment_paths or None,
            )
            return {"accepted": True, "queued": True, "task_id": task.id}

        async def job() -> None:
            await autonomy.run_turn_immediate(
                user_message,
                client_id=client_id,
                attachments=attachment_meta or None,
            )

        asyncio.create_task(job())
        return {"accepted": True, "queued": False}

    @app.get("/api/autonomy")
    async def api_autonomy_status(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        autonomy: AutonomyEngine = request.app.state.autonomy
        return await autonomy.status()

    @app.get("/api/tasks")
    async def api_tasks_list(request: Request, limit: int = 50) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        autonomy: AutonomyEngine = request.app.state.autonomy
        tasks = await autonomy.queue.list_tasks(limit=min(100, max(1, limit)))
        return {"tasks": tasks, "enabled": settings.autonomy_enabled}

    @app.post("/api/tasks")
    async def api_tasks_enqueue(body: TaskBody, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        if not body.message.strip():
            raise HTTPException(400, "empty message")
        if not settings.autonomy_enabled:
            raise HTTPException(400, "Autonomy is disabled")
        autonomy: AutonomyEngine = request.app.state.autonomy
        task = await autonomy.enqueue_manual(body.message.strip(), client_id=body.client_id)
        return {"accepted": True, "task_id": task.id}

    @app.post("/api/tasks/clear")
    async def api_tasks_clear(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        autonomy: AutonomyEngine = request.app.state.autonomy
        removed = await autonomy.queue.clear_completed()
        return {"ok": True, "removed": removed}

    @app.post("/api/watchdog/inspect")
    async def api_watchdog_inspect(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        watchdog: JobWatchdog = request.app.state.watchdog
        interventions = await watchdog.inspect_once()
        return {
            "ok": True,
            "interventions": interventions,
            "running_jobs": len(request.app.state.sessions.running_jobs()),
        }

    @app.get("/api/soul")
    async def api_soul_get(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        soul: SoulStore = request.app.state.soul
        content = soul.load()
        return {
            "path": str(soul.path),
            "content": content,
            "empty": soul.is_empty(content),
        }

    @app.delete("/api/soul")
    async def api_soul_delete(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        soul: SoulStore = request.app.state.soul
        soul.reset()
        bus: EventBus = request.app.state.bus
        await bus.publish({"type": "soul_updated", "source": "reset"})
        return {"ok": True, "path": str(soul.path)}

    @app.get("/api/conversation")
    async def api_conversation_get(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        memory: MemoryStore = request.app.state.memory
        return {
            "turns": len(memory.recent_turns_list()),
            "enabled": settings.conversation_memory_enabled,
        }

    @app.delete("/api/conversation")
    async def api_conversation_delete(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        memory: MemoryStore = request.app.state.memory
        memory.clear()
        bus: EventBus = request.app.state.bus
        await bus.publish({"type": "memory_cleared", "source": "reset"})
        return {"ok": True}

    @app.get("/api/usage")
    async def api_usage_get(request: Request, scope: str = "session") -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        if not settings.token_usage_enabled:
            return {"enabled": False}
        usage: TokenUsageStore = request.app.state.usage
        snapshot = usage.snapshot(scope)
        snapshot["enabled"] = True
        return snapshot

    @app.get("/api/jobs")
    async def api_jobs(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        sessions: SessionManager = request.app.state.sessions
        return {"jobs": sessions.list_jobs()}

    @app.get("/api/jobs/{job_id}")
    async def api_job_detail(job_id: str, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        sessions: SessionManager = request.app.state.sessions
        detail = sessions.get_job_dict(job_id)
        if not detail:
            raise HTTPException(404, "job not found")
        return {"job": detail}

    @app.post("/api/jobs/clear")
    async def api_jobs_clear(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        sessions: SessionManager = request.app.state.sessions
        removed = sessions.clear_jobs(include_running=False)
        return {"ok": True, "removed": removed}

    @app.get("/api/user-memory")
    async def api_user_memory_get(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        pm: PersistentMemoryStore = request.app.state.persistent_memory
        return {"content": pm.load_user(), "path": str(pm.user_path)}

    @app.delete("/api/user-memory")
    async def api_user_memory_delete(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        pm: PersistentMemoryStore = request.app.state.persistent_memory
        pm.reset_user()
        return {"ok": True}

    @app.get("/api/agent-memory")
    async def api_agent_memory_get(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        pm: PersistentMemoryStore = request.app.state.persistent_memory
        return {"content": pm.load_memory(), "path": str(pm.memory_path)}

    @app.delete("/api/agent-memory")
    async def api_agent_memory_delete(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        pm: PersistentMemoryStore = request.app.state.persistent_memory
        pm.reset_memory()
        return {"ok": True}

    @app.get("/api/checkpoints")
    async def api_checkpoints_list(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        cp: CheckpointManager = request.app.state.checkpoint_manager
        return {"checkpoints": cp.list_checkpoints(), "enabled": settings.checkpoints_enabled}

    @app.post("/api/checkpoints/{checkpoint_id}/rollback")
    async def api_checkpoint_rollback(checkpoint_id: str, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        cp: CheckpointManager = request.app.state.checkpoint_manager
        result = cp.rollback(checkpoint_id)
        await request.app.state.bus.publish({"type": "checkpoint_rollback", **result})
        return result

    @app.get("/api/cron")
    async def api_cron_list(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        store: CronStore = request.app.state.cron_store
        return {"jobs": [j.to_dict() for j in store.list_jobs()], "enabled": settings.cron_enabled}

    @app.post("/api/cron")
    async def api_cron_create(body: CronBody, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        store: CronStore = request.app.state.cron_store
        job = CronJob(
            id=uuid.uuid4().hex,
            name=body.name,
            prompt=body.prompt,
            schedule_type=body.schedule_type,
            cron_expr=body.cron_expr,
            skill_names=body.skill_names or [],
        )
        store.upsert(job)
        return {"job": job.to_dict()}

    @app.delete("/api/cron/{job_id}")
    async def api_cron_delete(job_id: str, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        store: CronStore = request.app.state.cron_store
        return {"ok": store.delete(job_id)}

    @app.post("/api/batch")
    async def api_batch_start(body: BatchBody, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        if not settings.batch_enabled:
            raise HTTPException(400, "Batch processing disabled")
        runner: BatchRunner = request.app.state.batch_runner
        batch_id = runner.start(body.prompts)
        return {"batch_id": batch_id}

    @app.get("/api/batch")
    async def api_batch_list(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        runner: BatchRunner = request.app.state.batch_runner
        return {"batches": runner.list_batches()}

    @app.get("/api/batch/{batch_id}")
    async def api_batch_status(batch_id: str, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        runner: BatchRunner = request.app.state.batch_runner
        status = runner.status(batch_id)
        if not status:
            raise HTTPException(404, "batch not found")
        return status

    @app.get("/api/hooks")
    async def api_hooks_list(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        reg: HookRegistry = request.app.state.hook_registry
        return {"hooks": reg.list_hooks(), "enabled": settings.hooks_enabled}

    @app.post("/api/hooks")
    async def api_hooks_create(body: HookBody, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        reg: HookRegistry = request.app.state.hook_registry
        hook = body.model_dump()
        return {"hook": reg.add(hook)}

    @app.delete("/api/hooks/{hook_id}")
    async def api_hooks_delete(hook_id: str, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        reg: HookRegistry = request.app.state.hook_registry
        return {"ok": reg.delete(hook_id)}

    @app.post("/api/jobs/{job_id}/stop")
    async def api_job_stop(job_id: str, request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        sessions: SessionManager = request.app.state.sessions
        result = await sessions.terminate_job(job_id)
        if not result.get("ok"):
            raise HTTPException(400, result.get("error", "stop failed"))
        return result

    return app


app = create_app()
