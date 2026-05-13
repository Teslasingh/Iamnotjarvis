from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    app.state.settings = settings
    app.state.bus = bus
    app.state.sessions = sessions
    app.state.memory = memory
    yield


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
    async def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return JSONResponse({"error": "UI not built; missing static/index.html"}, status_code=500)
        return FileResponse(index_path)

    @app.get("/favicon.ico")
    async def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket, replay: bool = False) -> None:
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
        if not body.message.strip():
            raise HTTPException(400, "empty message")
        settings: Settings = request.app.state.settings
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
        sessions: SessionManager = request.app.state.sessions
        return {"jobs": sessions.list_jobs()}

    return app


app = create_app()
