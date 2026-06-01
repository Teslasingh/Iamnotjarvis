from __future__ import annotations

from typing import Dict, List, Optional, Set

from friday.agent.tools import TOOL_DEFINITIONS
from friday.config import Settings

TOOLSETS: Dict[str, Set[str]] = {
    "shell": {
        "run_shell",
        "start_shell_job",
        "get_shell_job",
        "list_shell_jobs",
        "stop_shell_job",
        "restart_friday",
    },
    "filesystem": {
        "read_file",
        "write_file",
        "replace_in_file",
        "list_dir",
        "resolve_path",
        "move_path",
        "delete_path",
        "make_dir",
        "inspect_file",
    },
    "web": {"web_search", "http_request"},
    "code": {"search_code", "validate_python", "sqlite_query"},
    "memory": {"remember_soul", "get_token_usage"},
    "output": {"write_output", "deliver_output"},
    "meta": {"get_system_info"},
    "skills": {"load_skill"},
    "delegation": {"delegate_task"},
    "code_exec": {"execute_code"},
}

ROLE_TOOLSETS: Dict[str, Set[str]] = {
    "explore": {"filesystem", "web", "code", "meta", "skills", "output"},
    "execute": set(TOOLSETS.keys()) - {"delegation"},
    "verify": {"shell", "filesystem", "code", "meta", "skills", "output"},
    "delegate": {"filesystem", "web", "code", "meta", "skills", "output", "shell"},
}

ALL_TOOLSET_NAMES = frozenset(TOOLSETS.keys())


def _parse_enabled_toolsets(settings: Settings) -> Set[str]:
    raw = (settings.friday_toolsets_enabled or "").strip()
    if not raw:
        return set(ALL_TOOLSET_NAMES)
    names = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return names & ALL_TOOLSET_NAMES or set(ALL_TOOLSET_NAMES)


def _allowed_tool_names(
    settings: Settings,
    *,
    platform: str = "web",
    role: Optional[str] = None,
    extra_toolsets: Optional[List[str]] = None,
) -> Set[str]:
    del platform  # reserved for future channel-specific toolsets
    enabled = _parse_enabled_toolsets(settings)
    if role and role in ROLE_TOOLSETS:
        enabled &= ROLE_TOOLSETS[role]
    if extra_toolsets:
        for name in extra_toolsets:
            n = name.strip().lower()
            if n in ALL_TOOLSET_NAMES:
                enabled.add(n)
    names: Set[str] = set()
    for ts in enabled:
        names.update(TOOLSETS.get(ts, set()))
    if not settings.allow_shell:
        names -= TOOLSETS["shell"]
    if not settings.soul_enabled:
        names.discard("remember_soul")
    if not settings.token_usage_enabled:
        names.discard("get_token_usage")
    if not settings.friday_self_restart_enabled:
        names.discard("restart_friday")
    if not settings.skills_enabled:
        names.discard("load_skill")
    if not settings.delegate_enabled:
        names.discard("delegate_task")
    if not settings.code_exec_enabled:
        names.discard("execute_code")
    return names


def resolve_tools(
    settings: Settings,
    *,
    platform: str = "web",
    role: Optional[str] = None,
    extra_toolsets: Optional[List[str]] = None,
) -> List[Dict]:
    allowed = _allowed_tool_names(
        settings, platform=platform, role=role, extra_toolsets=extra_toolsets
    )
    return [t for t in TOOL_DEFINITIONS if t.get("function", {}).get("name") in allowed]
