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

from friday.agent.orchestrator import run_orchestrated_turn
from friday.agent.memory import MemoryStore
from friday.agent.soul import SoulStore
from friday.config import Settings, get_settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.llm.soul_update import maybe_update_soul
from friday.runtime.files import FileRegistry, guess_mime, is_path_under, is_preview_image, normalize_rel_path, sanitize_filename
from friday.runtime.sessions import SessionManager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    bus = EventBus(ring_max=settings.event_ring_max)
    workdir = settings.agent_workdir
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    (_workdir_path(settings) / settings.upload_dir).mkdir(parents=True, exist_ok=True)
    (_workdir_path(settings) / settings.output_dir).mkdir(parents=True, exist_ok=True)
    sessions = SessionManager(bus=bus, default_cwd=workdir, settings=settings)
    memory = MemoryStore(recent_turns=settings.memory_recent_turns)
    memory.clear()
    soul = SoulStore(workdir_path / "soul.md")
    soul.load()
    app.state.settings = settings
    app.state.bus = bus
    app.state.sessions = sessions
    app.state.memory = memory
    app.state.soul = soul
    registry_path = workdir_path / ".friday" / "file_registry.json"
    registry = FileRegistry(registry_path, workdir_path)
    registry.register_existing_paths(workdir_path, [settings.upload_dir, settings.output_dir])
    app.state.file_registry = registry
    try:
        yield
    finally:
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
        memory: MemoryStore = request.app.state.memory
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
        bus: EventBus = request.app.state.bus
        sessions: SessionManager = request.app.state.sessions
        memory: MemoryStore = request.app.state.memory
        registry: FileRegistry = request.app.state.file_registry
        soul: SoulStore = request.app.state.soul
        llm = LLMClient(settings=settings)
        user_message = body.message.strip()
        client_id = body.client_id

        attachment_meta: List[Dict[str, Any]] = []
        for rel_path in body.attachments or []:
            path = _validate_attachment_path(settings, rel_path)
            meta = registry.register(path, _workdir_path(settings), name=path.name)
            meta["path"] = normalize_rel_path(meta.get("path") or rel_path)
            attachment_meta.append(meta)

        async def job() -> None:
            try:
                memory_context = memory.build_context()
                session_uploads = memory.recent_attachments()
                soul_context = ""
                if settings.soul_enabled:
                    soul_context = soul.build_context(settings.soul_max_context_chars)
                    if soul_context:
                        await bus.publish(
                            {
                                "type": "soul_loaded",
                                "chars": len(soul_context),
                            }
                        )
                if memory_context:
                    await bus.publish(
                        {
                            "type": "memory_loaded",
                            "turns": len(memory.recent_turns_list()),
                        }
                    )
                reply, outputs = await run_orchestrated_turn(
                    user_message,
                    llm=llm,
                    bus=bus,
                    sessions=sessions,
                    workdir=settings.agent_workdir,
                    allow_shell=settings.allow_shell,
                    max_steps=settings.max_agent_steps,
                    settings=settings,
                    memory_context=memory_context,
                    soul_context=soul_context,
                    attachments=attachment_meta,
                    session_uploads=session_uploads,
                    file_registry=registry,
                    soul_store=soul,
                    client_id=client_id,
                )
                memory.append_turn(user_message, reply, attachments=attachment_meta or None)
                await bus.publish({"type": "memory_saved", "client_id": client_id})
                if settings.soul_enabled and settings.soul_auto_update:
                    asyncio.create_task(
                        maybe_update_soul(
                            soul,
                            llm=llm,
                            settings=settings,
                            bus=bus,
                            user_message=user_message,
                            assistant_reply=reply,
                            client_id=client_id,
                        )
                    )
                await bus.publish(
                    {
                        "type": "chat_complete",
                        "reply": reply,
                        "client_id": client_id,
                        "outputs": outputs,
                    }
                )
            except Exception as exc:
                logger.exception("chat job failed")
                await bus.publish({"type": "chat_error", "error": str(exc), "client_id": client_id})

        asyncio.create_task(job())
        return {"accepted": True}

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
