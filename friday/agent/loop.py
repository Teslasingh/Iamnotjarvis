from __future__ import annotations

import asyncio
import json
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 — Any used by agent extras

from friday.agent.execution_intent import implies_host_execution
from friday.agent.checkpoints import CheckpointManager
from friday.agent.mistakes import TurnMistakeLog, record_tool_outcome, shell_run_failed
from friday.agent.persistent_memory import PersistentMemoryStore
from friday.agent.skills import SkillEntry
from friday.agent.toolsets import resolve_tools
from friday.agent.tools import ToolContext, execute_tool
from friday.agent.soul import SoulStore
from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.llm.query_expansion import expand_user_query
from friday.runtime.files import FileRegistry
from friday.runtime.sessions import SessionManager


SYSTEM_PROMPT = """You are Friday, the core orchestrator for the Iamnotjarvis platform on this machine.
Your mission: autonomously process user requests, expand them with engineering judgment, execute via tools,
and safely modify the repository (friday_repo_root from get_system_info) when required.
Operate like a calm Jarvis-like operator: talk less, do more. Be precise, composed, observant,
and action-first. Use dry, light humor only when it naturally fits, never when work is failing
or the user needs a direct answer.

Orchestrator blueprint (internal; do not recite):
- Expand requests beyond literal wording: infer edge cases, security, performance, error handling, and architectural impact.
- Use multi-phase workflows only when complexity warrants it; simple tasks execute directly in one pass.
- Self-heal: diagnose blockers, try alternate approaches, cross-check against repo files, validate before claiming success.
- Apply production-grade repo updates: minimal diffs, match existing conventions, no stray scratch artifacts.

Communication style:
- Prefer tool use and concrete action over explaining what you might do.
- Keep replies short: usually 1-4 sentences unless the user asks for detail.
- Do not over-apologize, lecture, or produce long plans for obvious tasks.
- Ask a question only when you cannot safely infer the next step.
- Report results plainly: what changed, what passed, what remains.
- If the user asks for a task, execute it like an assistant, not a chatbot. Minimal theatrics, maximum utility.
- Never say "I can't do that from here" until you have tried the appropriate tool or confirmed the process lacks permission.

You have host-level tools: run_shell, start_shell_job, stop_shell_job, get_shell_job, list_shell_jobs,
read_file, write_file, write_output, replace_in_file, inspect_file, deliver_output, list_dir, resolve_path, move_path, delete_path, make_dir,
get_system_info, search_code, validate_python, get_token_usage, restart_friday, web_search, http_request, sqlite_query, remember_soul.
They run on this machine with inherited environment variables (paths, conda, npm, SDKs, secrets-in-env — same visibility as Friday's process).

Shell behavior (approximates your normal CLI; not a full interactive TTY):
- Unix jobs use bash invoked with --login when FRIDAY_UNIX_LOGIN_SHELL=true so profile PATH loads like a desktop shell session.
- Windows defaults to PowerShell (utf-16 -EncodedCommand) so PS syntax works as you type it; FRIDAY_WINDOWS_SHELL=cmd restores cmd.exe semantics.

Tool policy:
- For actionable host requests, use tools first. Do not give manual instructions as the first response.
- When the user asks what is running (tmux, processes, docker, services) or says "on my system", run diagnostic shell commands immediately and report live output — not a tutorial.
- DEFAULT start_shell_job for anything the user runs as an application — dev servers, watchers, GUIs, games,
  notebooks, bots — and OMIT the timeout field entirely so jobs run indefinitely until exit or explicit stop_shell_job.
  Never pass tiny timeouts like 5–120 seconds unless the task is deliberately bounded (e.g. benchmark with hard cap).
  Parallelism is encouraged: unrelated commands get separate start_shell_jobs.
- ONLY use stop_shell_job after the user wants a job ended; resolve job_id from list_shell_jobs or get_shell_job.
- Use run_shell for finite work: installs, builds, lint, quick tests, one-shot probes, short scripts that must exit.
- Poll get_shell_job when you need streamed output summaries; UI also shows tails live.
- When a job exits, job_finished fires with outcome and tails; run_shell still returns stdout/stderr in its tool payload.
- Use read_file/write_file/replace_in_file/list_dir/make_dir/move_path/delete_path for filesystem operations anywhere the OS allows.
- For new code paths, prefer write_file then start_shell_job so the process stays visible in the jobs sidebar.
- If a user gives a path that does not exist, do not immediately ask for clarification. Use resolve_path, list parent directories, and choose the nearest plausible match when confidence is high.
- Prefer replace_in_file for small edits; write_file for new files or full rewrites.
- Use web_search, http_request, sqlite_query appropriately.
- Only delete paths when explicitly asked.
- Use load_skill when a task matches a skill in the catalog; follow its instructions.
- Use delegate_task for parallel isolated sub-agents when fan-out helps.
- Use remember_soul when the user asks you to remember a preference, habit, or standing instruction.
  Also call remember_soul (section learnings) after you diagnose and fix a mistake — capture root cause
  and the working approach so future sessions avoid the same failure.
  Soul memory is curated long-term storage (soul.md), not a chat transcript — save durable facts only.

Learning from mistakes:
- Soul memory includes a Learnings section with past failures; read it every turn and do not repeat those errors.
- When a command, path, tool choice, or edit fails then you recover, save the lesson via remember_soul (learnings).
- Prefer a different strategy after a failure; never loop the same failing tool call without new evidence.

Repository update protocol:
- Call get_system_info before any repo edit for friday_package_dir, friday_repo_root, agent_workdir, and soul_path.
- Use search_code to locate symbols/strings, read_file for context, replace_in_file for surgical edits, write_file for new files.
- After editing friday/*.py, run validate_python on changed files; fix errors before claiming success.
- Avoid orphan scratch files in the workdir; integrate changes into the existing architecture.
- Do not claim git cleanliness or success without tool-verified validation output.
- Tell the user to restart python -m friday after Python source changes, or use restart_friday when FRIDAY_SELF_RESTART_ENABLED=true.

Self-modification (soul-driven, reactive):
- soul.md holds durable intent (Preferences, Behaviors, Learnings, Environment, Self); friday/ source is the implementation.
- Workflow: get_system_info → search_code → read_file → replace_in_file → validate_python → remember_soul (Self section).
- Edit your own code when the user explicitly asks to change Friday/Iamnotjarvis, or when soul Behaviors/Preferences require
  new tools, config defaults, UI changes, or system-prompt behavior that memory alone cannot enforce.
- Changes must land in friday_repo_root via file tools — never paste code-only answers without writing files.
- Record applied source changes in soul Self section via remember_soul when appropriate.
- Do not store secrets in soul; do not manually edit .friday/ registry.

Token usage:
- When the user asks about tokens, usage, or cost, call get_token_usage and report the numbers plainly — do not guess.

Workspace file management (agent_workdir):
- uploads/ — user uploads (attached in chat)
- outputs/ — generated deliverables for download
- .friday/ — internal download registry; avoid manual edits
- Use move_path to rename, reorganize, or shift files/folders within the workdir.
- Use delete_path only when the user explicitly asks to delete.
- Prefer list_dir before bulk cleanup; keep one canonical copy when deduplicating uploads.

Self-healing protocol:
- Before changing existing code, inspect relevant files with read_file/list_dir.
- On blocker (error, invalid path, missing dep): diagnose root cause, try a materially different approach,
  cross-examine changes against existing repo files, then retry within the step budget.
- After code changes, run validation when possible (import, syntax, test, lint, or startup check).
- If validation fails, read stderr, patch, and retry — do not report success until verified or budget exhausted.
- If one method fails, try an equivalent native command or alternate tool before giving up.
- If permission blocks the goal, try a non-admin path, then report the exact blocker.
- Do not repeat the same tool call with the same arguments unless new evidence justifies it.
- Do not pretend success; state exactly what remains unverified.

When authoring interactive Python CLI scripts (write_file then run_shell), prefer sturdy I/O similar to classical terminal runners:
- Use try/except (and optionally finally) around main work; surface errors with explicit print messages.
- For user prompts, separate print(prompt, flush=True, end="") from the following input() line so line-buffered consoles behave reliably.
- For numeric input, coerce with int() / float() after input(); flush=True on prints helps watchers and background jobs show live output.

run_shell responses include an outcome object with execution_signals and suspect_failure: if exit_ok is false or suspect_failure is true, treat the run as unsuccessful and fix or rerun after reading stderr.

Upload-driven task protocol:
- When the user attaches files, inspect them first with inspect_file (binary/unknown) or read_file (plain text).
- Generate code/scripts under the workdir; write deliverables to the configured output_dir when producing files for the user.
- Install dependencies with run_shell before running generated code (pip, npm, etc.).
- After every execution, check outcome.suspect_failure; if true, read stderr, patch code or install missing deps, and rerun within the step budget.
- Include key stdout in your final reply when the user asked for textual results.
- Call deliver_output for any file the user should download or preview (charts, exports, reports).

Text extraction and output-file rules:
- NEVER claim text was extracted, converted, OCR'd, or saved unless you used tools and verified stdout or file contents.
- For images, scans, or PDFs: inspect_file first, then run_shell with Python (pdfplumber, pymupdf, easyocr, etc.) or native OCR tools.
- Tesseract is optional and already configured on this host when needed (web-ui/.env TESSERACT_CMD). Do NOT install, PATH-check, or re-verify Tesseract unless the user explicitly asks about OCR.
- When the user asks for a .txt or any downloadable output: prefer write_output(filename, content) which saves UTF-8 under output_dir and registers the download automatically. Alternatively write_file under output_dir then deliver_output in the same turn.
- Save text outputs as UTF-8 with normalized line endings. On Windows, write_output includes a UTF-8 BOM so Notepad opens files correctly.
- On follow-up messages ("give me the txt file"), use session upload paths from context — do NOT ask to re-upload unless list_dir shows the file is gone.
- If you extracted text in a prior turn but did not save it, re-read the source upload from disk and write the output file now.

For coding tasks, prefer read_file/inspect_file/replace_in_file/write_file/write_output/run_shell/start_shell_job/deliver_output.
For actionable requests, use tools before explaining; do not end a turn with only a plan or manual instructions.
Think Jarvis with a keyboard, not a committee meeting.

Autonomous operation (when enabled by the platform):
- You may receive [Autonomous continuation] or [Autonomous job follow-up] messages without a live user prompt.
- Treat these as real work: use tools, finish the task, or state clearly if no action is needed.
- Job follow-ups: review shell output; fix failures, deliver outputs, or restart services as appropriate.
- If a job follow-up is for a diagnostic/probe command (Test-Path, where, version checks) that already succeeded, reply in one sentence and stop — do not spawn more verification commands.
- Continuations: pick up where the prior turn stopped; do not repeat completed steps.
- A watchdog monitors all shell jobs: runaway processes, print loops, and hung jobs are stopped automatically.
- If repeated tool calls stall (same call failing), stop retrying the same approach and try something different."""


def _requires_tool_use(user_message: str, assistant_text: str) -> bool:
    text = user_message.lower()
    if implies_host_execution(user_message):
        return True
    informational_prefixes = (
        "what is ",
        "why ",
        "explain ",
        "how does ",
        "how do i ",
        "tell me ",
    )
    if text.strip().startswith(informational_prefixes):
        return False
    if text.strip().startswith("what are ") and not implies_host_execution(user_message):
        return False

    action_terms = (
        "show",
        "list",
        "ls ",
        "dir ",
        "update",
        "upgrade",
        "modify",
        "patch",
        "install",
        "uninstall",
        "remove",
        "delete",
        "clean",
        "clear",
        "fix",
        "run",
        "execute",
        "launch",
        "start",
        "build",
        "edit",
        "change",
        "make",
        "test",
        "debug",
        "deploy",
        "compile",
        "configure",
        "implement",
        "refactor",
        "server",
        "kill",
        "stop",
        "terminate",
        "close",
        "open",
        "create",
        "write",
        "read",
        "check",
        "scan",
        "search",
        "find",
        "move",
        "copy",
        "rename",
        "organize",
        "cleanup",
        "download",
        "extract",
        "convert",
        "export",
        "save",
        "give",
        "get",
        "send",
    )
    host_terms = (
        "file",
        "folder",
        "directory",
        "windows",
        "app",
        "apps",
        "machine",
        "system",
        "drive",
        "user",
        "users",
        "c:\\",
        "powershell",
        "cmd",
        "shell",
    )
    refusal_terms = (
        "can't push",
        "can't do",
        "cannot do",
        "from here",
        "do it yourself",
        "open settings",
    )
    self_mod_phrases = (
        "change your",
        "update yourself",
        "modify your",
        "patch your",
        "your source",
        "your code",
        "friday source",
        "system prompt",
    )
    token_phrases = (
        "token",
        "tokens",
        "token usage",
        "how many tokens",
        "usage stats",
    )

    has_action = any(term in text for term in action_terms)
    has_host_target = any(term in text for term in host_terms)
    has_self_mod = any(phrase in text for phrase in self_mod_phrases)
    has_token_query = any(phrase in text for phrase in token_phrases)
    refused = any(term in assistant_text.lower() for term in refusal_terms)
    return (
        refused
        or has_self_mod
        or has_token_query
        or has_action
        or has_host_target
    )


def _tool_signature(name: str, args: Dict[str, Any]) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def _shell_command_fingerprint(name: str, args: Dict[str, Any]) -> str:
    if name not in {"run_shell", "start_shell_job"}:
        return ""
    command = str(args.get("command") or "").strip().lower()
    if not command:
        return ""
    return " ".join(command.split())


def _planning_language(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "i will ",
        "i'll ",
        "you should ",
        "next i would ",
        "steps:",
        "step 1",
        "here's what i would do",
    )
    return any(marker in lowered for marker in markers)


def _build_user_message(user_message: str, attachments: Optional[List[Dict[str, Any]]]) -> str:
    if not attachments:
        return user_message
    lines = ["User attached files (saved on disk):"]
    for item in attachments:
        name = item.get("name") or item.get("path") or "file"
        path = item.get("path") or name
        size = item.get("size")
        mime = item.get("mime") or "unknown"
        size_text = f"{size:,} bytes" if isinstance(size, int) else "unknown size"
        lines.append(f"- {path} ({size_text}, {mime}) — use inspect_file or read_file")
    lines.append(f"User request: {user_message}")
    return "\n".join(lines)


def _merge_uploads(
    current: Optional[List[Dict[str, Any]]],
    session: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in (session or []) + (current or []):
        path = str(item.get("path") or "")
        if path:
            merged[path] = item
    return list(merged.values())


def _build_session_upload_context(uploads: List[Dict[str, Any]]) -> str:
    if not uploads:
        return ""
    lines = [
        "Session uploads still on disk (use these paths; do not ask the user to re-upload unless missing):"
    ]
    for item in uploads:
        path = item.get("path") or item.get("name") or "unknown"
        name = item.get("name") or path
        mime = item.get("mime") or "unknown"
        lines.append(f"- {path} ({name}, {mime})")
    return "\n".join(lines)


def _shell_run_failed(result: str) -> bool:
    return shell_run_failed(result)


async def _chat_with_retries(
    llm: LLMClient,
    messages: List[Dict[str, Any]],
    bus: EventBus,
    settings: Settings,
    tools: List[Dict[str, Any]],
    *,
    source: str = "agent_step",
) -> Dict[str, Any]:
    attempts = max(1, settings.llm_retries)
    timeout = max(5, settings.llm_timeout_seconds)
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(
                        llm.chat_with_tools,
                        messages=messages,
                        tools=tools,
                        source=source,
                    ),
                ),
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
            await bus.publish(
                {
                    "type": "llm_retry",
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "error": str(exc),
                }
            )
            if attempt < attempts:
                await asyncio.sleep(min(2.0, 0.4 * attempt))

    raise last_error or RuntimeError("LLM call failed")


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
    soul_context: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    session_uploads: Optional[List[Dict[str, Any]]] = None,
    file_registry: Optional[FileRegistry] = None,
    soul_store: Optional[SoulStore] = None,
    client_id: Optional[str] = None,
    effective_message: Optional[str] = None,
    task_brief: str = "",
    role_prompt: str = "",
    prior_agent_context: str = "",
    skip_expansion: bool = False,
    max_steps_override: Optional[int] = None,
    role: Optional[str] = None,
    extra_toolsets: Optional[List[str]] = None,
    context_files_context: str = "",
    skills_catalog: str = "",
    checkpoint_manager: Optional[CheckpointManager] = None,
    skills: Optional[Dict[str, SkillEntry]] = None,
    persistent_memory: Optional[PersistentMemoryStore] = None,
    hook_runner: Any = None,
    delegate_runner: Any = None,
    code_exec_runner: Any = None,
    autonomy_turn_source: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], TurnMistakeLog]:
    logger = logging.getLogger(__name__)
    step_limit = max_steps_override if max_steps_override is not None else max_steps
    mistake_log = TurnMistakeLog()

    if skip_expansion:
        resolved_message = (effective_message or user_message).strip()
    else:
        resolved_message, _expansion = await expand_user_query(
            user_message,
            llm=llm,
            settings=settings,
            bus=bus,
            memory_context=memory_context,
            soul_context=soul_context,
            attachments=attachments,
            client_id=client_id,
        )
    tools = resolve_tools(settings, role=role, extra_toolsets=extra_toolsets)
    ctx = ToolContext(
        bus=bus,
        sessions=sessions,
        workdir=workdir,
        allow_shell=allow_shell,
        settings=settings,
        file_registry=file_registry,
        soul_store=soul_store,
        usage_store=llm.usage_store,
        client_id=client_id,
        checkpoint_manager=checkpoint_manager,
        skills=skills,
        persistent_memory=persistent_memory,
        hook_runner=hook_runner,
        delegate_runner=delegate_runner,
        code_exec_runner=code_exec_runner,
        autonomy_turn_source=autonomy_turn_source,
    )
    composed_message = _build_user_message(resolved_message, attachments)
    upload_context = _build_session_upload_context(_merge_uploads(attachments, session_uploads))
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if context_files_context:
        messages.append({"role": "system", "content": context_files_context})
    if skills_catalog:
        messages.append({"role": "system", "content": skills_catalog})
    if role_prompt:
        messages.append({"role": "system", "content": role_prompt})
    if task_brief:
        messages.append({"role": "system", "content": task_brief})
    if prior_agent_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Prior sub-agent findings in this workflow (authoritative context):\n\n"
                    f"{prior_agent_context}"
                ),
            }
        )
    if soul_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Soul memory (persistent learnings from prior sessions). "
                    "Follow preferences and conventions here; do not recite unless relevant. "
                    "Treat Learnings as mistakes to avoid — never repeat those failures. "
                    "As orchestrator, honor soul Behaviors that imply repo or platform changes: "
                    "call get_system_info, read affected files, apply minimal diffs under friday_package_dir, "
                    "validate, and record in soul Self when appropriate.\n\n"
                    f"{soul_context}"
                ),
            }
        )
    if memory_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Recent conversation memory (loaded from disk when persistence is enabled). "
                    "Distinct from soul.md — this is recent chat continuity, not curated long-term facts. "
                    "Use for continuity when relevant, but do not repeat it unless asked.\n\n"
                    f"{memory_context}"
                ),
            }
        )
    if upload_context:
        messages.append({"role": "system", "content": upload_context})
    messages.append({"role": "user", "content": composed_message})

    await bus.publish({"type": "llm_turn_start", "user": composed_message[:2000]})

    final_text = ""
    tool_use_nudges = 0
    max_tool_nudges = 3
    tool_counts: Dict[str, int] = {}
    shell_fingerprints: Dict[str, int] = {}
    strategy_pivot_pending = False
    shell_failure_pending = False
    for step in range(step_limit):
        if strategy_pivot_pending:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Repeated tool calls detected. Stop retrying the same command or path. "
                        "Use a materially different strategy, tool, or approach now."
                    ),
                }
            )
            strategy_pivot_pending = False
        if shell_failure_pending:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Last run_shell failed (nonzero exit or suspect_failure). "
                        "Read stderr, cross-examine your changes against existing repo files, "
                        "fix code or install missing dependencies, then rerun."
                    ),
                }
            )
            shell_failure_pending = False

        await bus.publish({"type": "agent_step", "step": step})
        llm_source = "subagent" if role_prompt else "agent_step"
        try:
            assistant = await _chat_with_retries(
                llm, messages, bus, settings, tools, source=llm_source
            )
        except Exception as exc:
            await bus.publish({"type": "llm_error", "error": str(exc)})
            mistake_log.record(f"LLM error: {exc}")
            return f"LLM error: {exc}", ctx.delivered_outputs, mistake_log

        content = assistant.get("content") or ""
        tool_calls = assistant.get("tool_calls")
        should_retry_with_tools = (
            not tool_calls
            and tool_use_nudges < max_tool_nudges
            and (
                _requires_tool_use(resolved_message, content)
                or _planning_language(content)
                or bool(attachments)
            )
        )
        messages.append(
            {
                "role": "assistant",
                "content": content,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            }
        )

        if should_retry_with_tools:
            tool_use_nudges += 1
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You answered an actionable host-machine request without tools. "
                        "Expand implicit requirements (edge cases, validation, repo impact) then use tools now "
                        "to inspect or execute. Do not provide manual instructions first. "
                        "If a command fails because of missing permissions, report the exact command and blocker briefly."
                    ),
                }
            )
            await bus.publish({"type": "tool_use_required_retry"})
            continue

        if content:
            final_text = content
            await bus.publish({"type": "assistant_delta", "text": content[:8000]})

        if not tool_calls:
            await bus.publish({"type": "llm_turn_end"})
            return final_text or "(no content)", ctx.delivered_outputs, mistake_log

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            tool_id = call.get("id", "")
            safe_args = args if isinstance(args, dict) else {}
            signature = _tool_signature(name, safe_args)
            tool_counts[signature] = tool_counts.get(signature, 0) + 1
            shell_fp = _shell_command_fingerprint(name, safe_args)
            if shell_fp:
                shell_fingerprints[shell_fp] = shell_fingerprints.get(shell_fp, 0) + 1
                if shell_fingerprints[shell_fp] > 2:
                    strategy_pivot_pending = True
            if tool_counts[signature] > 2:
                result = json.dumps(
                    {
                        "error": "repeated_tool_call",
                        "message": (
                            "This exact tool call has already been tried twice. "
                            "Choose a different command, path, tool, or strategy."
                        ),
                    }
                )
                mistake_log.record(
                    f"Repeated failing tool call ({name}): {json.dumps(safe_args, ensure_ascii=False)[:240]}"
                )
                record_tool_outcome(mistake_log, name, safe_args, result)
                strategy_pivot_pending = True
                await bus.publish({"type": "agent_stall_detected", "tool": name})
            else:
                result = await execute_tool(ctx, name, safe_args)
                record_tool_outcome(mistake_log, name, safe_args, result)
                if name in {"run_shell", "start_shell_job", "get_shell_job"} and _shell_run_failed(result):
                    shell_failure_pending = True
            await bus.publish({"type": "tool_result", "tool": name, "result_preview": result[:2000]})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                }
            )

    logger.warning("agent max steps exceeded")
    mistake_log.step_budget_exhausted = True
    mistake_log.record("Agent step budget exhausted before task completion")
    await bus.publish({"type": "agent_max_steps"})
    return final_text or "Max agent steps reached.", ctx.delivered_outputs, mistake_log
