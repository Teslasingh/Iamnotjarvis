"""FastAPI application entry point.

Run with:
    python -m uvicorn app:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes_auth import router as auth_router
from api.routes_drafts import router as drafts_router
from api.routes_emails import router as emails_router
from api.routes_profile import router as profile_router
from api.routes_prompts import router as prompts_router
from api.routes_sync import router as sync_router
from config import BASE_DIR, ensure_directories, settings
from logging_config import setup_logging
import storage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    ensure_directories()
    storage.init_db()
    logger.info("%s started", settings.app_name)
    yield
    logger.info("%s shutting down", settings.app_name)


def create_app() -> FastAPI:
    application = FastAPI(title=settings.app_name, lifespan=lifespan)
    application.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @application.get("/")
    def index() -> FileResponse:
        return FileResponse(BASE_DIR / "templates" / "index.html")

    application.include_router(auth_router)
    application.include_router(profile_router)
    application.include_router(prompts_router)
    application.include_router(sync_router)
    application.include_router(emails_router)
    application.include_router(drafts_router)
    return application


app = create_app()
