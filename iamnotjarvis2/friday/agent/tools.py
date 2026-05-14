from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sqlite3
from dataclasses import dataclass
from difflib import get_close_matches
from html import unescape
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote_plus

import httpx

from friday.config import Settings
from friday.events.bus import EventBus
from friday.runtime.sessions import SessionManager


@dataclass
class ToolContext:
    bus: EventBus
    sessions: SessionManager
    workdir: str
    allow_shell: bool
    settings: Settings


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command on the host machine and wait for completion. "
                "Use this for system operations, package commands, scripts, and host inspection. "
                "The optional cwd may be any path the process can access."
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
                "Start a shell command in the background on the host machine. "
                "Use for long-running or parallel operations. Poll with get_shell_job."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Optional working directory; defaults to AGENT_WORKDIR"},
                    "timeout": {"type": "number", "description": "Optional timeout seconds"},
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
            "description": "Write UTF-8 text to any host path the app process can access. Parent directories are created.",
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
                "Use for small code updates instead of rewriting a whole file."
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
            "description": "Return basic host system information and configured working directory.",
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
    {
        "type": "function",
        "function": {
            "name": "run_legacy_codegen",
            "description": (
                "Run the legacy Python code-generation pipeline: generates a script from the "
                "prompt, executes it, retries on errors, may run pip install. Progress appears "
                "as codegen_* events on the event bus. Use for 'write a small Python program that…' tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What the generated program should do",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def _resolve_path(workdir: str, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (Path(workdir) / p).resolve()


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


async def execute_tool(ctx: ToolContext, name: str, arguments: Dict[str, Any]) -> str:
    await ctx.bus.publish({"type": "tool_call", "tool": name, "args": arguments})

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
        timeout = float(timeout_arg) if timeout_arg is not None else None
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
        return json.dumps(
            {
                "job_id": job.id,
                "command": job.command,
                "cwd": job.cwd,
                "status": job.status.value,
                "return_code": job.return_code,
                "stdout": "".join(job.stdout_buf)[-20_000:],
                "stderr": "".join(job.stderr_buf)[-20_000:],
            },
            ensure_ascii=False,
        )

    if name == "list_shell_jobs":
        return json.dumps({"jobs": ctx.sessions.list_jobs()}, ensure_ascii=False)

    if name == "read_file":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return json.dumps({"path": str(path), "content": text[:200_000]})
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "write_file":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        content = str(arguments.get("content", ""))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return json.dumps({"ok": True, "path": str(path), "bytes": len(content.encode("utf-8"))})
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "replace_in_file":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        old = str(arguments.get("old", ""))
        new = str(arguments.get("new", ""))
        count = _as_int(arguments.get("count"), 1, minimum=0, maximum=1_000_000)
        if not old:
            return json.dumps({"error": "old text must not be empty"})
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            occurrences = text.count(old)
            if occurrences == 0:
                return json.dumps({"error": "old text not found", "path": str(path)})
            replaced = text.replace(old, new) if count == 0 else text.replace(old, new, count)
            path.write_text(replaced, encoding="utf-8")
            return json.dumps(
                {
                    "ok": True,
                    "path": str(path),
                    "occurrences_found": occurrences,
                    "occurrences_replaced": occurrences if count == 0 else min(count, occurrences),
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

    if name == "delete_path":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        recursive = bool(arguments.get("recursive", False))
        try:
            if path.is_dir():
                if not recursive:
                    return json.dumps({"error": "directory deletion requires recursive=true"})
                shutil.rmtree(path)
            else:
                path.unlink()
            return json.dumps({"ok": True, "deleted": str(path)})
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "make_dir":
        path = _resolve_path(ctx.workdir, str(arguments.get("path", "")))
        try:
            path.mkdir(parents=True, exist_ok=True)
            return json.dumps({"ok": True, "path": str(path)})
        except OSError as exc:
            return json.dumps({"error": str(exc)})

    if name == "get_system_info":
        return json.dumps(
            {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "cwd": os.getcwd(),
                "agent_workdir": ctx.workdir,
                "home": str(Path.home()),
                "allow_shell": ctx.allow_shell,
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

    if name == "run_legacy_codegen":
        if not ctx.settings.azure_openai_api_key:
            return json.dumps({"error": "AZURE_OPENAI_API_KEY is not configured"})
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return json.dumps({"error": "empty prompt"})
        from friday.legacy.codegen import run_codegen_non_interactive

        await run_codegen_non_interactive(
            prompt=prompt,
            bus=ctx.bus,
            workdir=ctx.workdir,
            settings=ctx.settings,
        )
        return json.dumps({"ok": True, "message": "Legacy codegen finished; see event log for codegen_* events."})

    return json.dumps({"error": f"unknown tool {name}"})
