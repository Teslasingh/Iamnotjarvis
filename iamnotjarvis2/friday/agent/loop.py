from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from friday.agent.tools import TOOL_DEFINITIONS, ToolContext, execute_tool
from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.runtime.sessions import SessionManager


SYSTEM_PROMPT = """You are Friday, a calm Jarvis-like operator for this machine.
Operate like a capable assistant: talk less, do more. Be precise, composed, observant,
and action-first. Use dry, light humor only when it naturally fits, never when work is failing
or the user needs a direct answer.

Communication style:
- Prefer tool use and concrete action over explaining what you might do.
- Keep replies short: usually 1-4 sentences unless the user asks for detail.
- Do not over-apologize, lecture, or produce long plans for obvious tasks.
- Ask a question only when you cannot safely infer the next step.
- Report results plainly: what changed, what passed, what remains.
- If the user asks for a task, execute it like an assistant, not a chatbot. Minimal theatrics, maximum utility.

You have host-level tools: run_shell, start_shell_job, get_shell_job, list_shell_jobs,
read_file, write_file, replace_in_file, list_dir, resolve_path, delete_path, make_dir,
get_system_info, web_search, http_request, sqlite_query, run_legacy_codegen.
These tools operate on the real host machine with the same OS permissions as this Python process.

Tool policy:
- Use run_shell for commands, builds, package installs, tests, and host inspection.
- Use start_shell_job for long-running or parallel shell work; poll with get_shell_job and keep context across steps.
- Use read_file/write_file/replace_in_file/list_dir/make_dir/delete_path for filesystem operations anywhere the host OS allows.
- If a user gives a path that does not exist, do not immediately ask for clarification. Use resolve_path, list parent directories, and choose the nearest plausible match when confidence is high.
- Prefer replace_in_file for code updates when changing a small region; use write_file for new files or full rewrites.
- Use web_search for current public web information, http_request for direct URL/API access, and sqlite_query for local SQLite databases.
- Only delete files or directories when the user explicitly asks for deletion.

Self-correcting coding protocol:
- Before changing existing code, inspect relevant files.
- After code changes, run a reasonable validation command when possible (import check, test, lint, or app startup check).
- If validation fails, read the error, patch the code, and retry within the available step budget.
- Do not pretend success; if something cannot be validated, say exactly what remains unverified.

Use run_legacy_codegen only when the user wants the old template-driven Python program generator
(generate -> run -> retry with pip). For general coding tasks, prefer read_file/replace_in_file/write_file/run_shell.
When finished, respond with a short final summary and stop calling tools. Think Jarvis with a keyboard,
not a committee meeting."""


async def run_agent_turn(
    user_message: str,
    llm: LLMClient,
    bus: EventBus,
    sessions: SessionManager,
    workdir: str,
    allow_shell: bool,
    max_steps: int,
    settings: Settings,
    memory_context: str = "",
) -> str:
    logger = logging.getLogger(__name__)
    ctx = ToolContext(
        bus=bus,
        sessions=sessions,
        workdir=workdir,
        allow_shell=allow_shell,
        settings=settings,
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if memory_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Recent conversation memory, loaded from local storage. "
                    "Use it for continuity when relevant, but do not repeat it unless asked.\n\n"
                    f"{memory_context}"
                ),
            }
        )
    messages.append({"role": "user", "content": user_message})

    await bus.publish({"type": "llm_turn_start", "user": user_message[:2000]})

    final_text = ""
    for step in range(max_steps):
        await bus.publish({"type": "agent_step", "step": step})
        try:
            assistant = llm.chat_with_tools(messages=messages, tools=TOOL_DEFINITIONS)
        except Exception as exc:
            await bus.publish({"type": "llm_error", "error": str(exc)})
            return f"LLM error: {exc}"

        content = assistant.get("content") or ""
        tool_calls = assistant.get("tool_calls")
        messages.append(
            {
                "role": "assistant",
                "content": content,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }
        )

        if content:
            final_text = content
            await bus.publish({"type": "assistant_delta", "text": content[:8000]})

        if not tool_calls:
            await bus.publish({"type": "llm_turn_end"})
            return final_text or "(no content)"

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            tool_id = call.get("id", "")
            result = await execute_tool(ctx, name, args if isinstance(args, dict) else {})
            await bus.publish({"type": "tool_result", "tool": name, "result_preview": result[:2000]})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                }
            )

    logger.warning("agent max steps exceeded")
    await bus.publish({"type": "agent_max_steps"})
    return final_text or "Max agent steps reached."
