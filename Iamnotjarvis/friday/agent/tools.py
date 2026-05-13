from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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


async def execute_tool(ctx: ToolContext, name: str, arguments: Dict[str, Any]) -> str:
    await ctx.bus.publish({"type": "tool_call", "tool": name, "args": arguments})

    if name == "run_shell":
        if not ctx.allow_shell:
            return json.dumps({"error": "shell execution disabled by policy"})
        command = str(arguments.get("command", ""))
        cwd = arguments.get("cwd") or ctx.workdir
        result = await ctx.sessions.run_command(command, cwd=cwd, timeout=600.0)
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
        content = str(arguments.get("content", ""))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return json.dumps({"ok": True, "path": str(path), "bytes": len(content.encode("utf-8"))})
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
