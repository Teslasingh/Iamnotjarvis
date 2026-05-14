from __future__ import annotations

import asyncio
import hmac
import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from friday.agent.loop import run_agent_turn
from friday.agent.memory import MemoryStore
from friday.config import Settings, get_settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.runtime.sessions import SessionManager

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
AUTH_COOKIE = "friday_session"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    bus = EventBus(ring_max=settings.event_ring_max)
    workdir = settings.agent_workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)
    sessions = SessionManager(bus=bus, default_cwd=workdir)
    memory = MemoryStore(settings.memory_dir, recent_turns=settings.memory_recent_turns)
    memory.clear()
    app.state.settings = settings
    app.state.bus = bus
    app.state.sessions = sessions
    app.state.memory = memory
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
        llm = LLMClient(settings=settings)
        user_message = body.message.strip()

        async def job() -> None:
            try:
                memory_context = memory.build_context()
                if memory_context:
                    await bus.publish(
                        {
                            "type": "memory_loaded",
                            "turns": len(memory.recent_turns_list()),
                        }
                    )
                reply = await run_agent_turn(
                    user_message,
                    llm=llm,
                    bus=bus,
                    sessions=sessions,
                    workdir=settings.agent_workdir,
                    allow_shell=settings.allow_shell,
                    max_steps=settings.max_agent_steps,
                    settings=settings,
                    memory_context=memory_context,
                )
                memory.append_turn(user_message, reply)
                await bus.publish({"type": "memory_saved"})
                await bus.publish({"type": "chat_complete", "reply": reply})
            except Exception as exc:
                logger.exception("chat job failed")
                await bus.publish({"type": "chat_error", "error": str(exc)})

        asyncio.create_task(job())
        return {"accepted": True}

    @app.get("/api/jobs")
    async def api_jobs(request: Request) -> Dict[str, Any]:
        settings: Settings = request.app.state.settings
        if not _is_authenticated(request, settings):
            raise HTTPException(401, "Unauthorized")
        sessions: SessionManager = request.app.state.sessions
        return {"jobs": sessions.list_jobs()}

    return app


app = create_app()
