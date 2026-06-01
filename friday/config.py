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


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


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
    conversation_memory_enabled: bool
    conversation_memory_max_turns: int
    conversation_memory_max_context_chars: int
    conversation_memory_clear_on_logout: bool
    token_usage_enabled: bool
    token_usage_persist: bool
    token_usage_call_log_max: int
    friday_self_restart_enabled: bool
    autonomy_enabled: bool
    autonomy_poll_seconds: int
    autonomy_max_continuations: int
    autonomy_auto_continue: bool
    autonomy_job_followup: bool
    autonomy_queue_user_tasks: bool
    autonomy_watchdog_enabled: bool
    autonomy_watchdog_poll_seconds: int
    autonomy_job_max_runtime_seconds: int
    autonomy_job_stall_seconds: int
    autonomy_job_output_loop_repeats: int
    autonomy_continuation_stall_max: int
    autonomy_agent_stall_max: int
    friday_toolsets_enabled: str
    skills_enabled: bool
    friday_skills_dirs: str
    skills_max_catalog_chars: int
    skill_max_chars: int
    user_memory_enabled: bool
    agent_memory_enabled: bool
    context_files_enabled: bool
    context_files_max_chars: int
    refs_enabled: bool
    refs_max_file_chars: int
    refs_max_total_chars: int
    refs_allow_url_fetch: bool
    checkpoints_enabled: bool
    checkpoints_max_count: int
    checkpoints_max_file_bytes: int
    cron_enabled: bool
    cron_tick_seconds: int
    cron_nl_parse_enabled: bool
    cron_max_jobs: int
    delegate_enabled: bool
    delegate_max_parallel: int
    delegate_max_tasks_per_call: int
    delegate_subagent_max_steps: int
    code_exec_enabled: bool
    code_exec_max_seconds: int
    code_exec_max_tool_calls: int
    code_exec_allowed_tools: str
    hooks_enabled: bool
    hooks_webhook_timeout_seconds: int
    batch_enabled: bool
    batch_max_parallel: int
    batch_max_items: int


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
        conversation_memory_enabled=_env("CONVERSATION_MEMORY_ENABLED", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        conversation_memory_max_turns=_int_env(
            "CONVERSATION_MEMORY_MAX_TURNS",
            _int_env("MEMORY_RECENT_TURNS", 24),
        ),
        conversation_memory_max_context_chars=_int_env("CONVERSATION_MEMORY_MAX_CONTEXT_CHARS", 12000),
        conversation_memory_clear_on_logout=_env("CONVERSATION_MEMORY_CLEAR_ON_LOGOUT", "false").lower()
        in {"1", "true", "yes", "on", "y"},
        token_usage_enabled=_env("TOKEN_USAGE_ENABLED", "true").lower() in {"1", "true", "yes", "on", "y"},
        token_usage_persist=_env("TOKEN_USAGE_PERSIST", "true").lower() in {"1", "true", "yes", "on", "y"},
        token_usage_call_log_max=_int_env("TOKEN_USAGE_CALL_LOG_MAX", 200),
        friday_self_restart_enabled=_env("FRIDAY_SELF_RESTART_ENABLED", "false").lower()
        in {"1", "true", "yes", "on", "y"},
        autonomy_enabled=_env("AUTONOMY_ENABLED", "true").lower() in {"1", "true", "yes", "on", "y"},
        autonomy_poll_seconds=max(1, _int_env("AUTONOMY_POLL_SECONDS", 5)),
        autonomy_max_continuations=max(0, _int_env("AUTONOMY_MAX_CONTINUATIONS", 10)),
        autonomy_auto_continue=_env("AUTONOMY_AUTO_CONTINUE", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        autonomy_job_followup=_env("AUTONOMY_JOB_FOLLOWUP", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        autonomy_queue_user_tasks=_env("AUTONOMY_QUEUE_USER_TASKS", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        autonomy_watchdog_enabled=_env("AUTONOMY_WATCHDOG_ENABLED", "true").lower()
        in {"1", "true", "yes", "on", "y"},
        autonomy_watchdog_poll_seconds=max(5, _int_env("AUTONOMY_WATCHDOG_POLL_SECONDS", 15)),
        autonomy_job_max_runtime_seconds=_int_env("AUTONOMY_JOB_MAX_RUNTIME_SECONDS", 7200),
        autonomy_job_stall_seconds=_int_env("AUTONOMY_JOB_STALL_SECONDS", 300),
        autonomy_job_output_loop_repeats=max(5, _int_env("AUTONOMY_JOB_OUTPUT_LOOP_REPEATS", 25)),
        autonomy_continuation_stall_max=max(1, _int_env("AUTONOMY_CONTINUATION_STALL_MAX", 3)),
        autonomy_agent_stall_max=max(3, _int_env("AUTONOMY_AGENT_STALL_MAX", 8)),
        friday_toolsets_enabled=_env("FRIDAY_TOOLSETS_ENABLED"),
        skills_enabled=_bool_env("SKILLS_ENABLED", True),
        friday_skills_dirs=_env("FRIDAY_SKILLS_DIRS"),
        skills_max_catalog_chars=_int_env("SKILLS_MAX_CATALOG_CHARS", 4000),
        skill_max_chars=_int_env("FRIDAY_SKILL_MAX_CHARS", 24000),
        user_memory_enabled=_bool_env("USER_MEMORY_ENABLED", True),
        agent_memory_enabled=_bool_env("AGENT_MEMORY_ENABLED", True),
        context_files_enabled=_bool_env("CONTEXT_FILES_ENABLED", True),
        context_files_max_chars=_int_env("CONTEXT_FILES_MAX_CHARS", 12000),
        refs_enabled=_bool_env("REFS_ENABLED", True),
        refs_max_file_chars=_int_env("REFS_MAX_FILE_CHARS", 16000),
        refs_max_total_chars=_int_env("REFS_MAX_TOTAL_CHARS", 48000),
        refs_allow_url_fetch=_bool_env("REFS_ALLOW_URL_FETCH", False),
        checkpoints_enabled=_bool_env("CHECKPOINTS_ENABLED", True),
        checkpoints_max_count=_int_env("CHECKPOINTS_MAX_COUNT", 20),
        checkpoints_max_file_bytes=_int_env("CHECKPOINTS_MAX_FILE_BYTES", 2_000_000),
        cron_enabled=_bool_env("CRON_ENABLED", False),
        cron_tick_seconds=max(10, _int_env("CRON_TICK_SECONDS", 30)),
        cron_nl_parse_enabled=_bool_env("CRON_NL_PARSE_ENABLED", True),
        cron_max_jobs=_int_env("CRON_MAX_JOBS", 50),
        delegate_enabled=_bool_env("DELEGATE_ENABLED", True),
        delegate_max_parallel=max(1, _int_env("DELEGATE_MAX_PARALLEL", 3)),
        delegate_max_tasks_per_call=max(1, _int_env("DELEGATE_MAX_TASKS_PER_CALL", 5)),
        delegate_subagent_max_steps=_int_env("DELEGATE_SUBAGENT_MAX_STEPS", 16),
        code_exec_enabled=_bool_env("CODE_EXEC_ENABLED", False),
        code_exec_max_seconds=max(5, _int_env("CODE_EXEC_MAX_SECONDS", 30)),
        code_exec_max_tool_calls=max(1, _int_env("CODE_EXEC_MAX_TOOL_CALLS", 20)),
        code_exec_allowed_tools=_env(
            "CODE_EXEC_ALLOWED_TOOLS",
            "read_file,list_dir,search_code,run_shell",
        ),
        hooks_enabled=_bool_env("HOOKS_ENABLED", False),
        hooks_webhook_timeout_seconds=max(1, _int_env("HOOKS_WEBHOOK_TIMEOUT_SECONDS", 10)),
        batch_enabled=_bool_env("BATCH_ENABLED", False),
        batch_max_parallel=max(1, _int_env("BATCH_MAX_PARALLEL", 3)),
        batch_max_items=max(1, _int_env("BATCH_MAX_ITEMS", 500)),
    )
