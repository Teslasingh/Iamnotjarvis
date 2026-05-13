from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_azure_endpoint(value: str) -> str:
    if not value:
        return ""
    text = value.rstrip("/")
    parsed = urlparse(text)
    if not parsed.scheme:
        return "https://" + text
    return text


@dataclass(frozen=True)
class Settings:
    azure_openai_endpoint: str
    azure_openai_api_key: Optional[str]
    azure_openai_deployment_name: str
    azure_openai_api_version: str
    host: str
    port: int
    agent_workdir: str
    allow_shell: bool
    max_agent_steps: int
    event_ring_max: int
    memory_dir: str
    memory_recent_turns: int


@lru_cache
def get_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except ImportError:
        pass

    return Settings(
        azure_openai_endpoint=_normalize_azure_endpoint(_env("AZURE_OPENAI_ENDPOINT")),
        azure_openai_api_key=_env("AZURE_OPENAI_API_KEY") or None,
        azure_openai_deployment_name=_env("DEPLOYMENT_NAME") or _env("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_openai_api_version=_env("OPENAI_API_VERSION", "2024-12-01-preview"),
        host=_env("HOST", "0.0.0.0"),
        port=_int_env("PORT", 80),
        agent_workdir=os.path.abspath(_env("AGENT_WORKDIR") or os.getcwd()),
        allow_shell=_env("ALLOW_SHELL", "true").lower() in {"1", "true", "yes", "on", "y"},
        max_agent_steps=_int_env("MAX_AGENT_STEPS", 32),
        event_ring_max=_int_env("EVENT_RING_MAX", 500),
        memory_dir=os.path.abspath(_env("MEMORY_DIR") or os.path.join(os.getcwd(), "agent_memory")),
        memory_recent_turns=_int_env("MEMORY_RECENT_TURNS", 12),
    )
