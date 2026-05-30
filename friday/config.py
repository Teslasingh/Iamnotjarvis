from __future__ import annotations

import os
import secrets
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
    llm_timeout_seconds: int
    llm_retries: int
    event_ring_max: int
    memory_recent_turns: int
    friday_password: Optional[str]
    session_secret: str
    unix_login_shell: bool
    unix_kill_background_group: bool
    windows_shell: str
    upload_dir: str
    output_dir: str
    output_inline_max_bytes: int
    output_preview_image_max_bytes: int
    query_expansion_enabled: bool
    query_expansion_min_chars: int
    query_expansion_max_input_chars: int
    soul_enabled: bool
    soul_auto_update: bool
    soul_max_context_chars: int
    soul_max_file_chars: int
    task_analysis_enabled: bool
    multi_agent_enabled: bool
    multi_agent_max_subagents: int
    multi_agent_subagent_max_steps: int
    multi_agent_synthesis_max_chars: int


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
        llm_timeout_seconds=_int_env("LLM_TIMEOUT_SECONDS", 90),
        llm_retries=_int_env("LLM_RETRIES", 3),
        event_ring_max=_int_env("EVENT_RING_MAX", 500),
        memory_recent_turns=_int_env("MEMORY_RECENT_TURNS", 12),
        friday_password=_env("FRIDAY_PASSWORD") or None,
        session_secret=_env("SESSION_SECRET") or secrets.token_hex(32),
        unix_login_shell=_env("FRIDAY_UNIX_LOGIN_SHELL", "true").lower() in {"1", "true", "yes", "on", "y"},
        unix_kill_background_group=(
            _env("FRIDAY_UNIX_KILL_BACKGROUND_GROUP", "true").lower() in {"1", "true", "yes", "on", "y"}
        ),
        windows_shell=_env("FRIDAY_WINDOWS_SHELL", "powershell").strip().lower() or "powershell",
        upload_dir=_env("UPLOAD_DIR", "uploads"),
        output_dir=_env("OUTPUT_DIR", "outputs"),
        output_inline_max_bytes=_int_env("OUTPUT_INLINE_MAX_BYTES", 65536),
        output_preview_image_max_bytes=_int_env("OUTPUT_PREVIEW_IMAGE_MAX_BYTES", 5242880),
        query_expansion_enabled=_env("QUERY_EXPANSION_ENABLED", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        query_expansion_min_chars=_int_env("QUERY_EXPANSION_MIN_CHARS", 12),
        query_expansion_max_input_chars=_int_env("QUERY_EXPANSION_MAX_INPUT_CHARS", 1200),
        soul_enabled=_env("SOUL_ENABLED", "true").lower() in {"1", "true", "yes", "on", "y"},
        soul_auto_update=_env("SOUL_AUTO_UPDATE", "true").lower() in {"1", "true", "yes", "on", "y"},
        soul_max_context_chars=_int_env("SOUL_MAX_CONTEXT_CHARS", 8000),
        soul_max_file_chars=_int_env("SOUL_MAX_FILE_CHARS", 32000),
        task_analysis_enabled=_env("TASK_ANALYSIS_ENABLED", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        multi_agent_enabled=_env("MULTI_AGENT_ENABLED", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        multi_agent_max_subagents=_int_env("MULTI_AGENT_MAX_SUBAGENTS", 3),
        multi_agent_subagent_max_steps=_int_env("MULTI_AGENT_SUBAGENT_MAX_STEPS", 16),
        multi_agent_synthesis_max_chars=_int_env("MULTI_AGENT_SYNTHESIS_MAX_CHARS", 12000),
    )
