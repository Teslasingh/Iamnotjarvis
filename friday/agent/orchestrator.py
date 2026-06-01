from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from friday.agent.loop import run_agent_turn
from friday.agent.turn_context import AgentExtras
from friday.agent.mistakes import TurnMistakeLog
from friday.agent.soul import SoulStore
from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.llm.task_analysis import analyze_task, build_task_brief
from friday.runtime.files import FileRegistry
from friday.runtime.sessions import SessionManager

logger = logging.getLogger(__name__)

_SUBAGENT_BASE = (
    "You are a high-autonomy sub-agent in an Iamnotjarvis multi-agent workflow. "
    "Prior sub-agent findings are authoritative context — build on them, do not contradict without new evidence. "
    "Self-resolve blockers: invalid paths, missing dependencies, syntax errors, and failed commands. "
    "Debug your own work: read errors, cross-examine changes against existing repo files, try alternate "
    "approaches, and validate before reporting blocked."
)

_ROLE_PROMPTS: Dict[str, str] = {
    "explore": (
        f"{_SUBAGENT_BASE}\n\n"
        "Role: explore (Code Analyzer). Focus on read_file, list_dir, inspect_file, web_search, "
        "resolve_path, get_system_info. Map repo structure under friday_repo_root, read relevant source, "
        "identify edge cases and integration points. Do not mutate files or run destructive commands "
        "unless explicitly required."
    ),
    "execute": (
        f"{_SUBAGENT_BASE}\n\n"
        "Role: execute (Implementation Agent). Use the full tool set to implement the plan. "
        "Apply production-grade edits under friday_repo_root: read before write, minimal diffs, match "
        "existing conventions. Follow the self-healing protocol: inspect, change, validate, retry on failure."
    ),
    "verify": (
        f"{_SUBAGENT_BASE}\n\n"
        "Role: verify (QA / Self-heal Tester). Focus on run_shell, read_file, inspect_file. "
        "Run validation commands from the task brief success_criteria; if failures occur, patch and retry "
        "within your step budget before reporting blocked. Confirm what passed and what remains unverified."
    ),
}

_SYNTHESIS_SYSTEM = """You are Friday synthesizing sub-agent reports into one concise user-facing reply for Iamnotjarvis.
Be precise and action-first. 1-4 sentences unless detail is needed.
Summarize: what changed in the repo or filesystem, what validation passed, what remains unverified.
Do not mention sub-agents, orchestration, or internal workflow."""


def _subagent_failed(reply: str) -> bool:
    text = (reply or "").strip()
    if not text or text == "(no content)":
        return True
    lowered = text.lower()
    if lowered.startswith("llm error:"):
        return True
    if "max agent steps reached" in lowered:
        return True
    return False


def _merge_outputs(all_outputs: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for batch in all_outputs:
        for item in batch:
            key = str(item.get("id") or item.get("path") or item.get("name") or "")
            if key:
                seen[key] = item
            elif item not in seen.values():
                seen[str(id(item))] = item
    return list(seen.values())


async def _run_subagent(
    *,
    subtask: Dict[str, str],
    index: int,
    extra_kw: Optional[Dict[str, Any]] = None,
    user_message: str,
    expanded_query: str,
    task_brief: str,
    prior_context: str,
    llm: LLMClient,
    bus: EventBus,
    sessions: SessionManager,
    workdir: str,
    allow_shell: bool,
    settings: Settings,
    memory_context: str,
    soul_context: str,
    attachments: Optional[List[Dict[str, Any]]],
    session_uploads: Optional[List[Dict[str, Any]]],
    file_registry: Optional[FileRegistry],
    soul_store: Optional[SoulStore],
    client_id: Optional[str],
    retry_context: str = "",
) -> Tuple[str, List[Dict[str, Any]], bool, TurnMistakeLog]:
    kw = extra_kw or {}
    role = subtask.get("role", "execute")
    goal = subtask.get("goal", expanded_query)
    role_prompt = _ROLE_PROMPTS.get(role, _ROLE_PROMPTS["execute"])
    sub_brief = task_brief + f"\n\nSub-agent goal ({role}): {goal}"
    if retry_context:
        sub_brief += f"\n\nPrevious attempt failed:\n{retry_context}"

    await bus.publish(
        {
            "type": "subagent_start",
            "role": role,
            "index": index,
            "goal": goal[:500],
            "client_id": client_id,
        }
    )

    sub_message = f"{expanded_query}\n\nSub-task ({role}): {goal}"
    reply, outputs, mistakes = await run_agent_turn(
        user_message,
        llm=llm,
        bus=bus,
        sessions=sessions,
        workdir=workdir,
        allow_shell=allow_shell,
        max_steps=settings.max_agent_steps,
        settings=settings,
        memory_context=memory_context,
        soul_context=soul_context,
        attachments=attachments,
        session_uploads=session_uploads,
        file_registry=file_registry,
        soul_store=soul_store,
        client_id=client_id,
        effective_message=sub_message,
        task_brief=sub_brief,
        role_prompt=role_prompt,
        prior_agent_context=prior_context,
        skip_expansion=True,
        max_steps_override=settings.multi_agent_subagent_max_steps,
        role=role if role in {"explore", "execute", "verify"} else "execute",
        **kw,
    )

    failed = _subagent_failed(reply)
    await bus.publish(
        {
            "type": "subagent_complete",
            "role": role,
            "index": index,
            "failed": failed,
            "client_id": client_id,
        }
    )
    return reply, outputs, failed, mistakes


async def _synthesize_reply(
    llm: LLMClient,
    settings: Settings,
    user_message: str,
    reports: List[Dict[str, Any]],
) -> str:
    if not reports:
        return "Task completed."

    max_chars = settings.multi_agent_synthesis_max_chars
    lines = [f"User request: {user_message.strip()[:2000]}", "", "Sub-agent reports:"]
    for report in reports:
        role = report.get("role", "agent")
        goal = report.get("goal", "")
        reply = str(report.get("reply") or "")[:4000]
        lines.append(f"[{role}] goal: {goal}\n{reply}\n")

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[: max_chars - 20].rstrip() + "\n[... truncated ...]"

    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user", "content": context},
    ]

    try:
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(llm.chat, messages=messages, temperature=0.2, source="orchestrator_synthesis"),
            ),
            timeout=max(10, min(60, settings.llm_timeout_seconds)),
        )
    except Exception as exc:
        logger.warning("synthesis failed: %s", exc)
        last = reports[-1].get("reply") if reports else ""
        return str(last or "Task completed with partial results.")


async def run_orchestrated_turn(
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
    agent_extras: Optional[AgentExtras] = None,
) -> Tuple[str, List[Dict[str, Any]], TurnMistakeLog]:
    extra_kw = agent_extras.as_kwargs() if agent_extras else {}
    analysis = await analyze_task(
        user_message,
        llm=llm,
        settings=settings,
        bus=bus,
        memory_context=memory_context,
        soul_context=soul_context,
        attachments=attachments,
        client_id=client_id,
    )

    expanded_query = str(analysis.get("expanded_query") or user_message).strip()
    task_brief = build_task_brief(analysis)
    orchestrate = bool(analysis.get("orchestrate")) and settings.multi_agent_enabled
    subtasks: List[Dict[str, str]] = list(analysis.get("subtasks") or [])

    if not orchestrate or not subtasks:
        return await run_agent_turn(
            user_message,
            llm=llm,
            bus=bus,
            sessions=sessions,
            workdir=workdir,
            allow_shell=allow_shell,
            max_steps=max_steps,
            settings=settings,
            memory_context=memory_context,
            soul_context=soul_context,
            attachments=attachments,
            session_uploads=session_uploads,
            file_registry=file_registry,
            soul_store=soul_store,
            client_id=client_id,
            effective_message=expanded_query,
            task_brief=task_brief,
            skip_expansion=True,
            **extra_kw,
        )

    await bus.publish(
        {
            "type": "orchestration_start",
            "subtasks": len(subtasks),
            "complexity": analysis.get("complexity"),
            "client_id": client_id,
        }
    )

    prior_context = ""
    reports: List[Dict[str, Any]] = []
    all_outputs: List[List[Dict[str, Any]]] = []
    turn_mistakes = TurnMistakeLog()

    for index, subtask in enumerate(subtasks):
        reply, outputs, failed, mistakes = await _run_subagent(
            subtask=subtask,
            index=index,
            extra_kw=extra_kw,
            user_message=user_message,
            expanded_query=expanded_query,
            task_brief=task_brief,
            prior_context=prior_context,
            llm=llm,
            bus=bus,
            sessions=sessions,
            workdir=workdir,
            allow_shell=allow_shell,
            settings=settings,
            memory_context=memory_context,
            soul_context=soul_context,
            attachments=attachments,
            session_uploads=session_uploads,
            file_registry=file_registry,
            soul_store=soul_store,
            client_id=client_id,
        )

        turn_mistakes.merge(mistakes)
        if mistakes.step_budget_exhausted:
            turn_mistakes.step_budget_exhausted = True

        if failed:
            await bus.publish(
                {
                    "type": "subagent_retry",
                    "role": subtask.get("role"),
                    "index": index,
                    "client_id": client_id,
                }
            )
            reply, outputs, _, retry_mistakes = await _run_subagent(
                subtask=subtask,
                index=index,
                extra_kw=extra_kw,
                user_message=user_message,
                expanded_query=expanded_query,
                task_brief=task_brief,
                prior_context=prior_context,
                llm=llm,
                bus=bus,
                sessions=sessions,
                workdir=workdir,
                allow_shell=allow_shell,
                settings=settings,
                memory_context=memory_context,
                soul_context=soul_context,
                attachments=attachments,
                session_uploads=session_uploads,
                file_registry=file_registry,
                soul_store=soul_store,
                client_id=client_id,
                retry_context=reply,
            )

            turn_mistakes.merge(retry_mistakes)
            if retry_mistakes.step_budget_exhausted:
                turn_mistakes.step_budget_exhausted = True

        role = subtask.get("role", "execute")
        goal = subtask.get("goal", "")
        reports.append({"role": role, "goal": goal, "reply": reply})
        all_outputs.append(outputs)
        prior_context += f"\n\n[{role}] {goal}\n{reply}"

    final_reply = await _synthesize_reply(llm, settings, user_message, reports)
    merged_outputs = _merge_outputs(all_outputs)

    await bus.publish(
        {
            "type": "orchestration_complete",
            "subtasks": len(subtasks),
            "client_id": client_id,
        }
    )
    return final_reply.strip() or "Task completed.", merged_outputs, turn_mistakes
