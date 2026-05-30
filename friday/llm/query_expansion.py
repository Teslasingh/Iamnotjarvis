from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient
from friday.llm.task_analysis import analyze_task


async def expand_user_query(
    message: str,
    llm: LLMClient,
    settings: Settings,
    bus: EventBus,
    memory_context: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    client_id: Optional[str] = None,
    soul_context: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Return (effective_message, metadata). Delegates to analyze_task."""
    original = message.strip()
    analysis = await analyze_task(
        original,
        llm=llm,
        settings=settings,
        bus=bus,
        memory_context=memory_context,
        soul_context=soul_context,
        attachments=attachments,
        client_id=client_id,
    )
    meta: Dict[str, Any] = {
        "original": original,
        "expanded": analysis.get("expanded_query", original),
        "applied": bool(analysis.get("applied")),
        "analysis": analysis,
    }
    return str(analysis.get("expanded_query") or original), meta
