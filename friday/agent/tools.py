from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import asyncio
from dataclasses import dataclass, field
from difflib import get_close_matches
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from friday.config import Settings
from friday.events.bus import EventBus
from friday.agent.checkpoints import CheckpointManager
from friday.agent.persistent_memory import PersistentMemoryStore, mirror_remember_soul
from friday.agent.skills import SkillEntry, load_skill_body
from friday.agent.soul import SoulStore
from friday.llm.usage import TokenUsageStore
from friday.runtime.files import (
    FileRegistry,
    guess_mime,
    is_path_under,
    is_preview_image,
    is_preview_text,
    is_text_extension,
    move_path_on_disk,
    sanitize_filename,
    sync_registry_after_move,
    workdir_relative,
    write_text_file,
)
from friday.runtime.shell_analysis import analyze_shell_streams
from friday.runtime.sessions import SessionManager


@dataclass
class ToolContext:
    bus: EventBus
    sessions: SessionManager
    workdir: str
    allow_shell: bool
    settings: Settings
    file_registry: Optional[FileRegistry] = None
    soul_store: Optional[SoulStore] = None
    usage_store: Optional[TokenUsageStore] = None
    client_id: Optional[str] = None
    delivered_outputs: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_manager: Optional[CheckpointManager] = None
    skills: Optional[Dict[str, SkillEntry]] = None
    persistent_memory: Optional[PersistentMemoryStore] = None
    hook_runner: Any = None
    delegate_runner: Any = None
    code_exec_runner: Any = None


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command on the host machine and wait for completion. "
                "Use this for system operations, package commands, scripts, and host inspection. "
                "The optional cwd may be any path the process can access. "
                "The JSON result includes an outcome field (execution_signals, suspect_failure). "
                "Treat suspect_failure true or nonzero return_code as failure even if stdout looks fine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Optional working directory; defaults to AGENT_WORKDIR"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_shell_job",
            "description": (
                "Start a shell command as a BACKGROUND job (parallel-friendly). "
                "Use this DEFAULT for runnable applications, scripts, servers, watchers, GUIs, and anything that "
                "should stay running until the user stops it — omit timeout so it runs indefinitely. "
                "Only pass timeout when you truly need an automatic cutoff. Poll with get_shell_job; UI shows live output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Optional working directory; defaults to AGENT_WORKDIR"},
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Omit entirely for run-until-exit (default for apps/servers). "
                            "Only set a positive number if you need a hard auto-kill (e.g. bounded tests). "
                            "0 or negative is treated as no timeout."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shell_job",
            "description": "Get status and accumulated output for a background shell job by job_id.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_shell_jobs",
            "description": "List known shell jobs and their statuses.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_shell_job",
            "description": (
                "Stop a running background shell job by job_id (terminate then force-kill after a delay). "
                "Unix background jobs may kill the entire process tree when FRIDAY_UNIX_KILL_BACKGROUND_GROUP=true. "
                "Use when the user asks to stop, close, or kill a running app; resolve job_id from list_shell_jobs or get_shell_job."
            ),
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_output",
            "description": (
                "Write a UTF-8 text deliverable under output_dir and register it for download/preview. "
                "Preferred for .txt/.csv exports when the user wants a saved output file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Output filename, e.g. extracted.txt",
                    },
                    "content": {"type": "string", "description": "Full UTF-8 text content to save"},
                    "label": {"type": "string", "description": "Optional display name in the UI"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_file",
            "description": (
                "Inspect any file (uploads, outputs, binaries). Returns size, extension, MIME guess, "
                "whether it is probably text, a text preview, or a hex preview for binary data. "
                "Use before read_file on unknown uploads."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deliver_output",
            "description": (
                "Register a generated output file for the user to download or preview in chat. "
                "Prefer writing deliverables under the configured outputs directory first. "
                "Call this when the user asked for an output file or when you produced a chart/report/export."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the output file under the workdir"},
                    "label": {"type": "string", "description": "Optional display name for the user"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from any host path the app process can access.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write UTF-8 text to any host path the app process can access. Parent directories are created. "
                "Read the target file first when updating existing code; prefer replace_in_file for small edits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": (
                "Safely edit an existing UTF-8 text file by replacing one exact string with another. "
                "Read the file first; use for surgical repo edits with minimal diffs instead of rewriting whole files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to replace"},
                    "new": {"type": "string", "description": "Replacement text"},
                    "count": {
                        "type": "integer",
                        "description": "Maximum replacements, default 1. Use 0 to replace all occurrences.",
                    },
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders in any host directory the app process can access.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_path",
            "description": (
                "Resolve a user-provided path and suggest nearest matching files/folders if it does not exist. "
                "Use this before asking the user to clarify misspelled paths like iamntjarvis vs Iamnotjarvis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "base": {"type": "string", "description": "Optional base directory for relative paths"},
                    "kind": {"type": "string", "description": "file, dir, or any"},
                    "max_suggestions": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_path",
            "description": (
                "Move or rename a file or directory within the workdir. "
                "If destination is an existing directory, the source is moved into it. "
                "Updates the download registry when tracked files move."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source file or directory path"},
                    "destination": {
                        "type": "string",
                        "description": "Target path or existing directory",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Replace destination if it already exists, default false",
                    },
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_path",
            "description": (
                "Delete a file or directory on the host machine. Directories require recursive=true. "
                "Use only when the user explicitly asks for deletion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "description": "Required for deleting directories"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_dir",
            "description": "Create a directory, including parent directories, anywhere the process can write.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": (
                "Return host system information including agent_workdir, friday_package_dir, friday_repo_root, "
                "and soul_path. Call before editing the Iamnotjarvis repository."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo Lite and return text results. Use for current public information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Maximum result snippets to return, default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an HTTP request to a URL and return status, headers, and a text body preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "description": "GET, POST, PUT, PATCH, DELETE; default GET"},
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                    "body": {"type": "string"},
                    "json": {"type": "object"},
                    "timeout": {"type": "number", "description": "Timeout seconds, default 30"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search for text or regex in the Iamnotjarvis repository (friday_repo_root from get_system_info). "
                "Returns matching file paths, line numbers, and line snippets. Use before editing repo source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Plain text or regex pattern to search"},
                    "path": {
                        "type": "string",
                        "description": "Optional subdirectory under friday_repo_root; defaults to repo root",
                    },
                    "regex": {"type": "boolean", "description": "Treat query as regex, default false"},
                    "max_results": {"type": "integer", "description": "Maximum matches to return, default 30"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_python",
            "description": (
                "Run python -m py_compile on one or more .py files to verify syntax after repo edits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Python file paths to validate",
                    }
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_token_usage",
            "description": (
                "Return Azure OpenAI token usage totals. Use when the user asks about tokens, usage, or cost. "
                "Do not guess — call this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "last_turn, session, or lifetime; default last_turn",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_friday",
            "description": (
                "Restart the Friday server process after editing friday/*.py. "
                "Only works when FRIDAY_SELF_RESTART_ENABLED=true. Use after validation passes."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_soul",
            "description": (
                "Save a durable preference, behavior, learning, environment note, or self-modification record to soul.md "
                "(persistent long-term memory). Use section 'learnings' for mistakes to avoid and fixes that worked. "
                "Use section 'self' for applied source changes requiring restart. "
                "Use when the user asks to remember something, when a standing instruction should persist across sessions, "
                "or after diagnosing and fixing a failure so future turns avoid the same error. "
                "Do not store secrets or one-off task details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The fact or instruction to remember"},
                    "section": {
                        "type": "string",
                        "description": "One of: preferences, behaviors, learnings, environment, self",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load full instructions for an agentskills.io skill by name. "
                "Call when the task matches a skill in the catalog before acting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name from the catalog"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                "Spawn parallel sub-agents with isolated context and restricted toolsets. "
                "Each task needs a prompt; optional role (explore, verify, delegate) and toolsets list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "role": {"type": "string"},
                                "toolsets": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["prompt"],
                        },
                    },
                    "share_context": {
                        "type": "boolean",
                        "description": "Share parent turn summary with each child",
                    },
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Run a short Python script in a sandboxed subprocess (CODE_EXEC_ENABLED). "
                "Use for multi-step tool workflows collapsed into one turn when enabled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sqlite_query",
            "description": (
                "Run a SQLite query against a database file accessible on the host. "
                "Use SELECT for reads; non-SELECT statements are committed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {"type": "string"},
                    "query": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}},
                    "readonly": {"type": "boolean", "description": "Open database read-only when true"},
                    "max_rows": {"type": "integer", "description": "Maximum rows returned for SELECT, default 100"},
                },
                "required": ["db_path", "query"],
            },
        },
    },
]


def _resolve_path(workdir: str, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (Path(workdir) / p).resolve()


def _friday_paths() -> Dict[str, Optional[str]]:
    import friday

    package_dir = Path(friday.__file__).resolve().parent
    repo_root: Optional[Path] = None
    for marker in ("setup.py", "pyproject.toml"):
        candidate = package_dir.parent / marker
        if candidate.is_file():
            repo_root = package_dir.parent
            break
    return {
        "friday_package_dir": str(package_dir),
        "friday_repo_root": str(repo_root) if repo_root else None,
    }


_SKIP_SEARCH_DIRS = {".git", "__pycache__", ".friday", "node_modules", ".venv", "venv"}


def _normalize_text_for_match(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _repo_edit_meta(path: Path) -> Dict[str, Any]:
    paths = _friday_paths()
    repo_root = paths.get("friday_repo_root")
    package_dir = paths.get("friday_package_dir")
    under_repo = bool(repo_root and is_path_under(Path(repo_root), path))
    requires_restart = False
    if package_dir and path.suffix.lower() == ".py":
        try:
            requires_restart = is_path_under(Path(package_dir), path)
        except ValueError:
            requires_restart = False
    validation_hint = None
    if path.suffix.lower() == ".py":
        validation_hint = f'python -m py_compile "{path}"'
    return {
        "under_friday_repo": under_repo,
        "requires_restart": requires_restart,
        "validation_hint": validation_hint,
    }


def _search_code_in_repo(
    repo_root: Path,
    query: str,
    *,
    subpath: Optional[str] = None,
    regex: bool = False,
    max_results: int = 30,
) -> List[Dict[str, Any]]:
    root = repo_root
    if subpath:
        root = (repo_root / subpath).resolve()
        if not is_path_under(repo_root, root):
            return []
    pattern = re.compile(query) if regex else None
    matches: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_SEARCH_DIRS]
        for filename in filenames:
            if len(matches) >= max_results:
                return matches
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() not in {
                ".py",
                ".md",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".html",
                ".js",
                ".css",
                ".toml",
                ".env",
            }:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, 1):
                if regex:
                    if not pattern or not pattern.search(line):
                        continue
                elif query not in line:
                    continue
                matches.append(
                    {
                        "path": str(file_path),
                        "line": line_no,
                        "snippet": line.strip()[:200],
                    }
                )
                if len(matches) >= max_results:
                    return matches
    return matches


def _replace_not_found_hint(text: str, old: str) -> Optional[str]:
    normalized = _normalize_text_for_match(text)
    old_norm = _normalize_text_for_match(old)
    idx = normalized.find(old_norm[: min(40, len(old_norm))])
    if idx < 0:
        return None
    start = max(0, idx - 80)
    end = min(len(normalized), idx + 120)
    return normalized[start:end]


def _strip_html(html_text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _as_int(value: Any, default: int, minimum: int = 1, maximum: int = 1000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _is_probably_text(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    text_chars = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b <= 126)
    return text_chars / len(sample) >= 0.85


def _output_preview_kind(settings: Settings, meta: Dict[str, Any]) -> str:
    mime = str(meta.get("mime") or "")
    name = str(meta.get("name") or "")
    size = int(meta.get("size") or 0)
    if is_preview_image(mime, name) and size <= settings.output_preview_image_max_bytes:
        return "image"
    if is_preview_text(mime, name) and size <= settings.output_inline_max_bytes:
        return "text"
    return "download"


async def _register_deliverable(ctx: ToolContext, path: Path, label: Optional[str] = None) -> Dict[str, Any]:
    workdir = Path(ctx.workdir).resolve()
    if not path.is_file() or not is_path_under(workdir, path):
        return {"error": "file not found or outside workdir", "path": str(path)}
    if not ctx.file_registry:
        return {"error": "file registry unavailable"}

    meta = ctx.file_registry.register(path, workdir, name=label or path.name)
    meta["preview_kind"] = _output_preview_kind(ctx.settings, meta)
    ctx.delivered_outputs.append(meta)
    await ctx.bus.publish(
        {
            "type": "output_ready",
            "client_id": ctx.client_id,
            "output": meta,
        }
    )
    return {"ok": True, **meta}


def _resolve_output_candidate(ctx: ToolContext, raw_path: str) -> Path:
    workdir = Path(ctx.workdir).resolve()
    path = _resolve_path(ctx.workdir, raw_path)
    if path.is_file():
        return path
    basename = Path(raw_path).name
    alt = (workdir / ctx.settings.output_dir / basename).resolve()
    if alt.is_file() and is_path_under(workdir, alt):
        return alt
    alt_safe = (workdir / ctx.settings.output_dir / sanitize_filename(basename)).resolve()
    if alt_safe.is_file() and is_path_under(workdir, alt_safe):
        return alt_safe
    return path


def _path_kind_matches(path: Path, kind: str) -> bool:
    if kind == "dir":
        return path.is_dir()
    if kind == "file":
        return path.is_file()
    return True


def _path_suggestions(target: Path, kind: str, max_suggestions: int) -> List[Dict[str, Any]]:
    parent = target.parent if target.parent.exists() else target.parent.parent
    if not parent.exists():
        return []
    try:
        children = list(parent.iterdir())
    except OSError:
        return []

    names = [child.name for child in children if _path_kind_matches(child, kind)]
    close_names = get_close_matches(target.name, names, n=max_suggestions, cutoff=0.35)
    close = [parent / name for name in close_names]
    if not close:
        normalized_target = re.sub(r"[^a-z0-9]", "", target.name.lower())
        scored = []
        for child in children:
            if not _path_kind_matches(child, kind):
                continue
            normalized_child = re.sub(r"[^a-z0-9]", "", child.name.lower())
            if normalized_target and (normalized_target in normalized_child or normalized_child in normalized_target):
                scored.append(child)
        close = scored[:max_suggestions]
    return [
        {
            "name": path.name,
            "path": str(path),
            "type": "dir" if path.is_dir() else "file",
        }
        for path in close
    ]


async def _maybe_checkpoint(ctx: ToolContext, tool: str, path: Path) -> None:
    if ctx.checkpoint_manager and ctx.settings.checkpoints_enabled and path.is_file():
        ctx.checkpoint_manager.snapshot_before(tool, [path])


async def execute_tool(ctx: ToolContext, name: str, arguments: Dict[str, Any]) -> str:
    await ctx.bus.publish({"type": "tool_call", "tool": name, "args": arguments})
    if ctx.hook_runner:
        deny = await ctx.hook_runner.before_tool_call(name, arguments)
        if deny:
            return json.dumps({"error": "hook_denied", "message": deny})

    if name == "run_shell":
        if not ctx.allow_shell:
            return json.dumps({"error": "shell execution disabled by policy"})
        command = str(arguments.get("command", ""))
        cwd = arguments.get("cwd") or ctx.workdir
        result = await ctx.sessions.run_command(command, cwd=cwd, timeout=600.0)
        return json.dumps(result, ensure_ascii=False)

    if name == "start_shell_job":
        if not ctx.allow_shell:
            return json.dumps({"error": "shell execution disabled by policy"})
        command = str(arguments.get("command", ""))
        cwd = arguments.get("cwd") or ctx.workdir
        timeout_arg = arguments.get("timeout")
        # Models often guess a numeric timeout ("safety rail") which kills servers/GUIs seconds later.
        # Only honor explicit positive timeouts; 0/false/absent runs until the process exits naturally.
        timeout: Optional[float] = None
        if timeout_arg is not None:
            try:
                t = float(timeout_arg)
            except (TypeError, ValueError):
                t = -1.0
            if t > 0:
                timeout = t
        job = await ctx.sessions.spawn(command, cwd=cwd, timeout=timeout)
        return json.dumps(
            {
                "job_id": job.id,
                "command": job.command,
                "cwd": job.cwd,
                "status": job.status.value,
                "return_code": job.return_code,
            },
            ensure_ascii=False,
        )

    if name == "get_shell_job":
        job_id = str(arguments.get("job_id", ""))
        job = ctx.sessions.get_job(job_id)
        if not job:
            return json.dumps({"error": "job not found", "job_id": job_id})
        out = "".join(job.stdout_buf)
        err = "".join(job.stderr_buf)
        outcome = analyze_shell_streams(out, err, job.return_code)
        return json.dumps(
            {
                "job_id": job.id,
                "command": job.command,
                "cwd": job.cwd,
                "status": job.status.value,
                "return_code": job.return_code,
                "stdout": out[-20_000:],
                "stderr": err[-20_000:],
                "outcome": outcome,
            },
            ensure_ascii=False,
        )

    if name == "list_shell_jobs":
        return json.dumps({"jobs": ctx.sessions.list_jobs()}, ensure_ascii=False)

    if name == "stop_shell_job":
        if not ctx.allow_shell:
            return json.dumps({"error": "shell execution disabled by policy"})
        job_id = str(arguments.get("job_id", ""))
        result = await ctx.sessions.terminate_job(job_id)
        return json.dumps(result, ensure_ascii=False)

    if name == "write_output":
        raw_name = sanitize_filename(str(arguments.get("filename", "output.txt")))
        if not Path(raw_name).suffix:
            raw_name = f"{raw_name}.txt"
        out_path = (Path(ctx.workdir) / ctx.settings.output_dir / raw_name).resolve()
        content = str(arguments.get("content", ""))
        label = str(arguments.get("label")).strip() if arguments.get("label") else raw_name
        try:
            if not is_path_under(Path(ctx.workdir).resolve(), out_path):
                return json.dumps({"error": "invalid output path"})
            byte_count = write_text_file(
                out_path,
                content,
                utf8_bom=FileRegistry.default_utf8_bom(),
            )
            result = await _register_deliverable(ctx, out_path, label)
            if result.get("ok"):
                result["bytes"] = byte_count
            return json.dumps(result, ensure_ascii=False)
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "inspect_file":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        try:
            stat = path.stat()
            sample = path.read_bytes()[:8192]
            probably_text = _is_probably_text(sample)
            mime = guess_mime(path)
            payload: Dict[str, Any] = {
                "path": str(path),
                "size": stat.st_size,
                "extension": path.suffix.lower(),
                "mime": mime,
                "is_probably_text": probably_text,
            }
            if probably_text:
                text = sample.decode("utf-8", errors="replace")
                payload["text_preview"] = text[:4000]
            else:
                payload["hex_preview"] = sample[:256].hex()
            return json.dumps(payload, ensure_ascii=False)
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "deliver_output":
        raw_path = str(arguments.get("path", ""))
        label = arguments.get("label")
        label_str = str(label).strip() if label else None
        path = _resolve_output_candidate(ctx, raw_path)
        result = await _register_deliverable(ctx, path, label_str)
        return json.dumps(result, ensure_ascii=False)

    if name == "read_file":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return json.dumps({"path": str(path), "content": text[:200_000]})
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "write_file":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        await _maybe_checkpoint(ctx, "write_file", path)
        content = str(arguments.get("content", ""))
        workdir = Path(ctx.workdir).resolve()
        try:
            if is_text_extension(path):
                byte_count = write_text_file(path, content, utf8_bom=FileRegistry.default_utf8_bom())
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                byte_count = len(content.encode("utf-8"))
            rel = workdir_relative(workdir, path)
            meta = _repo_edit_meta(path.resolve())
            return json.dumps(
                {
                    "ok": True,
                    "path": rel,
                    "absolute_path": str(path.resolve()),
                    "bytes": byte_count,
                    **meta,
                },
                ensure_ascii=False,
            )
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "replace_in_file":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        await _maybe_checkpoint(ctx, "replace_in_file", path)
        old = str(arguments.get("old", ""))
        new = str(arguments.get("new", ""))
        count = _as_int(arguments.get("count"), 1, minimum=0, maximum=1_000_000)
        if not old:
            return json.dumps({"error": "old text must not be empty"})
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            text = _normalize_text_for_match(raw_text)
            old_norm = _normalize_text_for_match(old)
            new_norm = _normalize_text_for_match(new)
            occurrences = text.count(old_norm)
            if occurrences == 0:
                hint = _replace_not_found_hint(raw_text, old)
                payload: Dict[str, Any] = {"error": "old text not found", "path": str(path)}
                if hint:
                    payload["nearby_context"] = hint
                return json.dumps(payload)
            replaced = text.replace(old_norm, new_norm) if count == 0 else text.replace(old_norm, new_norm, count)
            path.write_text(replaced, encoding="utf-8")
            meta = _repo_edit_meta(path.resolve())
            return json.dumps(
                {
                    "ok": True,
                    "path": str(path),
                    "occurrences_found": occurrences,
                    "occurrences_replaced": occurrences if count == 0 else min(count, occurrences),
                    **meta,
                }
            )
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "list_dir":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", ".")))
        try:
            entries = []
            for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:500]:
                try:
                    stat = child.stat()
                    size = stat.st_size
                except OSError:
                    size = None
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "type": "dir" if child.is_dir() else "file",
                        "size": size,
                    }
                )
            return json.dumps({"path": str(path), "entries": entries})
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "resolve_path":
        base = str(arguments.get("base") or ctx.workdir)
        raw_path = str(arguments.get("path", ""))
        kind = str(arguments.get("kind", "any")).lower()
        if kind not in {"file", "dir", "any"}:
            kind = "any"
        max_suggestions = _as_int(arguments.get("max_suggestions"), 5, maximum=25)
        path = _resolve_path(base, raw_path)
        exists = path.exists() and _path_kind_matches(path, kind)
        return json.dumps(
            {
                "input": raw_path,
                "resolved": str(path),
                "exists": exists,
                "type": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
                "suggestions": [] if exists else _path_suggestions(path, kind, max_suggestions),
            },
            ensure_ascii=False,
        )

    if name == "move_path":
        workdir = Path(ctx.workdir).resolve()
        src = _resolve_path(ctx.workdir, str(arguments.get("source", "")))
        if src.is_file():
            await _maybe_checkpoint(ctx, "move_path", src)
        dest = _resolve_path(ctx.workdir, str(arguments.get("destination", "")))
        overwrite = bool(arguments.get("overwrite", False))
        try:
            old_rel = workdir_relative(workdir, src)
            was_file = src.is_file()
            final = move_path_on_disk(src, dest, workdir, overwrite=overwrite)
            sync_registry_after_move(
                ctx.file_registry,
                src,
                final,
                workdir,
                was_file=was_file,
            )
            return json.dumps(
                {
                    "ok": True,
                    "source": old_rel,
                    "destination": workdir_relative(workdir, final),
                    "absolute_path": str(final),
                },
                ensure_ascii=False,
            )
        except (OSError, ValueError) as exc:
            return json.dumps({"error": str(exc)})

    if name == "delete_path":
        workdir = Path(ctx.workdir).resolve()
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        if path.is_file():
            await _maybe_checkpoint(ctx, "delete_path", path)
        recursive = bool(arguments.get("recursive", False))
        try:
            rel = workdir_relative(workdir, path)
            is_dir = path.is_dir()
            if is_dir:
                if not recursive:
                    return json.dumps({"error": "directory deletion requires recursive=true"})
                if ctx.file_registry:
                    ctx.file_registry.unregister_under(rel)
                shutil.rmtree(path)
            else:
                if ctx.file_registry:
                    ctx.file_registry.unregister_rel(rel)
                path.unlink()
            return json.dumps({"ok": True, "deleted": rel}, ensure_ascii=False)
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "make_dir":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        workdir = Path(ctx.workdir).resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            return json.dumps({"ok": True, "path": workdir_relative(workdir, path)}, ensure_ascii=False)
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "get_system_info":
        paths = _friday_paths()
        info: Dict[str, Any] = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
            "agent_workdir": ctx.workdir,
            "upload_dir": ctx.settings.upload_dir,
            "output_dir": ctx.settings.output_dir,
            "home": str(Path.home()),
            "allow_shell": ctx.allow_shell,
            "friday_package_dir": paths["friday_package_dir"],
            "friday_repo_root": paths["friday_repo_root"],
            "self_modify_note": (
                "Edits to friday/*.py take effect only after restarting the server (python -m friday)."
            ),
        }
        if ctx.soul_store:
            info["soul_path"] = str(ctx.soul_store.path)
        return json.dumps(info)

    if name == "search_code":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "empty query"})
        paths = _friday_paths()
        repo_root = paths.get("friday_repo_root")
        if not repo_root:
            return json.dumps({"error": "friday_repo_root not found"})
        subpath = str(arguments.get("path") or "").strip() or None
        regex = bool(arguments.get("regex", False))
        max_results = _as_int(arguments.get("max_results"), 30, maximum=100)
        try:
            matches = _search_code_in_repo(
                Path(repo_root),
                query,
                subpath=subpath,
                regex=regex,
                max_results=max_results,
            )
            return json.dumps(
                {
                    "repo_root": repo_root,
                    "query": query,
                    "match_count": len(matches),
                    "matches": matches,
                },
                ensure_ascii=False,
            )
        except re.error as exc:
            return json.dumps({"error": f"invalid regex: {exc}"})

    if name == "validate_python":
        raw_paths = arguments.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            return json.dumps({"error": "paths must be a non-empty array"})
        results: List[Dict[str, Any]] = []
        all_ok = True
        for raw in raw_paths[:20]:
            path = _resolve_path(ctx.workdir, str(raw))
            if path.suffix.lower() != ".py":
                results.append({"path": str(path), "ok": False, "error": "not a .py file"})
                all_ok = False
                continue
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", str(path)],
                capture_output=True,
                text=True,
            )
            ok = proc.returncode == 0
            if not ok:
                all_ok = False
            results.append(
                {
                    "path": str(path),
                    "ok": ok,
                    "stderr": (proc.stderr or proc.stdout or "").strip()[:2000],
                }
            )
        return json.dumps({"ok": all_ok, "results": results}, ensure_ascii=False)

    if name == "get_token_usage":
        if not ctx.settings.token_usage_enabled:
            return json.dumps({"error": "token usage tracking disabled"})
        if not ctx.usage_store:
            return json.dumps({"error": "token usage store unavailable"})
        scope = str(arguments.get("scope") or "last_turn").strip().lower()
        if scope not in {"last_turn", "session", "lifetime"}:
            scope = "last_turn"
        snapshot = ctx.usage_store.snapshot(scope)
        totals = snapshot.get("totals") or {}
        summary = (
            f"scope={snapshot.get('scope')}: "
            f"prompt={totals.get('prompt_tokens', 0)}, "
            f"completion={totals.get('completion_tokens', 0)}, "
            f"total={totals.get('total_tokens', 0)}"
        )
        snapshot["summary"] = summary
        return json.dumps(snapshot, ensure_ascii=False)

    if name == "restart_friday":
        if not ctx.settings.friday_self_restart_enabled:
            return json.dumps(
                {
                    "error": "self restart disabled",
                    "hint": "Set FRIDAY_SELF_RESTART_ENABLED=true to enable restart_friday",
                }
            )
        if not ctx.allow_shell:
            return json.dumps({"error": "shell execution disabled by policy"})
        cwd = ctx.workdir or os.getcwd()
        subprocess.Popen(
            [sys.executable, "-m", "friday"],
            cwd=cwd,
            env=os.environ.copy(),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0,
        )

        async def _delayed_exit() -> None:
            await asyncio.sleep(1.5)
            os._exit(0)

        asyncio.create_task(_delayed_exit())
        return json.dumps(
            {
                "ok": True,
                "message": "Friday restart scheduled; this process will exit shortly.",
                "cwd": cwd,
            }
        )

    if name == "web_search":
        query = str(arguments.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "empty query"})
        max_results = _as_int(arguments.get("max_results"), 5, maximum=20)
        url = "https://lite.duckduckgo.com/lite/?q=" + quote_plus(query)
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
            text = _strip_html(resp.text)
            # DuckDuckGo Lite is simple HTML; returning a compact text slice is robust
            # across minor markup changes and avoids extra parser dependencies.
            chunks = [chunk.strip() for chunk in re.split(r"\s{2,}", text) if chunk.strip()]
            return json.dumps(
                {
                    "query": query,
                    "source": url,
                    "results": chunks[:max_results],
                    "text_preview": text[:8000],
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc), "query": query})

    if name == "http_request":
        method = str(arguments.get("method", "GET")).upper()
        url = str(arguments.get("url", "")).strip()
        if not url:
            return json.dumps({"error": "empty url"})
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return json.dumps({"error": "url must start with http:// or https://"})
        headers = arguments.get("headers") if isinstance(arguments.get("headers"), dict) else {}
        timeout = float(arguments.get("timeout") or 30.0)
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    content=arguments.get("body"),
                    json=arguments.get("json"),
                )
            return json.dumps(
                {
                    "status_code": resp.status_code,
                    "url": str(resp.url),
                    "headers": dict(resp.headers),
                    "text": resp.text[:100_000],
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc), "url": url})

    if name == "sqlite_query":
        db_path = _resolve_path(ctx.workdir, str(arguments.get("db_path", "")))
        query = str(arguments.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "empty query"})
        params = arguments.get("params")
        if not isinstance(params, list):
            params = []
        max_rows = _as_int(arguments.get("max_rows"), 100, maximum=1000)
        readonly = bool(arguments.get("readonly", False))
        uri = f"file:{db_path}?mode=ro" if readonly else str(db_path)
        try:
            conn = sqlite3.connect(uri, uri=readonly, timeout=30)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(query, params)
            if query.lstrip().lower().startswith(("select", "pragma", "with")):
                rows = [dict(row) for row in cur.fetchmany(max_rows)]
                conn.close()
                return json.dumps({"db_path": str(db_path), "rows": rows, "row_count": len(rows)}, ensure_ascii=False)
            conn.commit()
            rowcount = cur.rowcount
            conn.close()
            return json.dumps({"db_path": str(db_path), "ok": True, "rowcount": rowcount})
        except Exception as exc:
            return json.dumps({"error": str(exc), "db_path": str(db_path)})

    if name == "remember_soul":
        if not ctx.soul_store:
            return json.dumps({"error": "soul memory unavailable"})
        if not ctx.settings.soul_enabled:
            return json.dumps({"error": "soul memory disabled"})
        text = str(arguments.get("text", "")).strip()
        if not text:
            return json.dumps({"error": "empty text"})
        section = str(arguments.get("section") or "learnings").strip()
        ok = ctx.soul_store.append_bullet(section, text)
        if not ok:
            return json.dumps({"error": "failed to save soul memory"})
        if ctx.persistent_memory:
            mirror_remember_soul(section, text, ctx.persistent_memory)
        await ctx.bus.publish({"type": "soul_updated", "source": "remember_soul", "client_id": ctx.client_id})
        return json.dumps(
            {
                "ok": True,
                "path": str(ctx.soul_store.path),
                "section": section,
                "text": text,
            },
            ensure_ascii=False,
        )

    if name == "load_skill":
        if not ctx.settings.skills_enabled:
            return json.dumps({"error": "skills disabled"})
        skill_name = str(arguments.get("name", "")).strip().lower()
        registry = ctx.skills or {}
        entry = registry.get(skill_name)
        if not entry:
            return json.dumps({"error": f"skill not found: {skill_name}"})
        body = load_skill_body(entry, ctx.settings.skill_max_chars)
        return json.dumps(
            {
                "name": entry.name,
                "description": entry.description,
                "instructions": body,
                "skill_dir": str(entry.skill_dir),
            },
            ensure_ascii=False,
        )

    if name == "delegate_task":
        if not ctx.delegate_runner:
            return json.dumps({"error": "delegation unavailable"})
        tasks = arguments.get("tasks")
        if not isinstance(tasks, list):
            return json.dumps({"error": "tasks must be a list"})
        share = bool(arguments.get("share_context", False))
        return await ctx.delegate_runner(tasks, share_context=share)

    if name == "execute_code":
        if not ctx.code_exec_runner:
            return json.dumps({"error": "execute_code unavailable"})
        code = str(arguments.get("code", ""))
        return await ctx.code_exec_runner(code)

    return json.dumps({"error": f"unknown tool {name}"})
