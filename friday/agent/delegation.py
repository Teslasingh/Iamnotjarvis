from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from friday.agent.loop import run_agent_turn
from friday.agent.mistakes import TurnMistakeLog
from friday.agent.soul import SoulStore
from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.runtime.files import FileRegistry
from friday.runtime.sessions import SessionManager

logger = logging.getLogger(__name__)


async def run_delegate_tasks(
    tasks: List[Dict[str, Any]],
    *,
    llm: LLMClient,
    bus: EventBus,
    sessions: SessionManager,
    workdir: str,
    allow_shell: bool,
    settings: Settings,
    file_registry: Optional[FileRegistry],
    soul_store: Optional[SoulStore],
    client_id: Optional[str],
    share_context: bool = False,
    parent_summary: str = "",
) -> str:
    if not settings.delegate_enabled:
        return json.dumps({"error": "delegation disabled"})
    capped = tasks[: settings.delegate_max_tasks_per_call]
    sem = asyncio.Semaphore(settings.delegate_max_parallel)

    async def _one(index: int, spec: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            prompt = str(spec.get("prompt") or "").strip()
            if not prompt:
                return {"index": index, "error": "empty prompt"}
            role = str(spec.get("role") or "delegate").strip().lower()
            toolsets = spec.get("toolsets")
            extra = toolsets if isinstance(toolsets, list) else None
            child_id = f"{client_id}:delegate:{index}" if client_id else f"delegate:{index}"
            prior = parent_summary if share_context else ""
            await bus.publish(
                {"type": "delegate_child_start", "index": index, "client_id": child_id}
            )
            try:
                reply, outputs, mistakes = await run_agent_turn(
                    prompt,
                    llm=llm,
                    bus=bus,
                    sessions=sessions,
                    workdir=workdir,
                    allow_shell=allow_shell,
                    max_steps=settings.delegate_subagent_max_steps,
                    settings=settings,
                    memory_context="",
                    soul_context="",
                    file_registry=file_registry,
                    soul_store=soul_store,
                    client_id=child_id,
                    role=role if role in {"explore", "verify", "delegate"} else "delegate",
                    extra_toolsets=extra,
                    prior_agent_context=prior,
                    skip_expansion=True,
                )
            except Exception as exc:
                logger.exception("delegate child failed")
                return {"index": index, "error": str(exc)}
            await bus.publish(
                {
                    "type": "delegate_child_complete",
                    "index": index,
                    "client_id": child_id,
                }
            )
            return {
                "index": index,
                "reply": reply,
                "outputs": outputs,
                "mistakes": {"entries": list(mistakes.entries)},
            }

    await bus.publish({"type": "delegate_start", "count": len(capped), "client_id": client_id})
    results = await asyncio.gather(*[_one(i, t) for i, t in enumerate(capped)])
    await bus.publish({"type": "delegate_complete", "client_id": client_id})
    return json.dumps({"results": results}, ensure_ascii=False)
