from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import partial
from typing import Any, Dict, Optional

from friday.agent.mistakes import TurnMistakeLog
from friday.agent.soul import SoulStore
from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient

logger = logging.getLogger(__name__)

SOUL_UPDATE_SYSTEM = """You maintain soul.md — Friday's persistent long-term memory across Iamnotjarvis sessions.

Output ONLY valid JSON. No markdown fences, no commentary.

Allowed shapes:
{"update": false}
{"update": true, "content": "<full merged soul.md markdown>"}

Save ONLY durable facts worth remembering next month:
- user preferences and standing instructions
- orchestration preferences (direct execution vs multi-step when user expresses a habit)
- recurring workflows, naming conventions, coding style
- repo conventions discovered (validation commands, test patterns, preferred tools)
- engineering standards the user expects (error handling, security habits, performance priorities)
- machine/environment quirks (paths, tools, OS habits)
- mistakes to avoid or lessons learned from repeated patterns
- standing instructions that imply tool, UI, or code behavior (Behaviors/Preferences)
- applied self-modifications under Self (what changed, restart required)

Do NOT save:
- one-off task details or transient command output
- repeated OCR/Tesseract install/PATH diagnostics (one consolidated Environment note is enough)
- full conversation transcripts
- secrets, passwords, API keys, tokens, or credentials
- greetings, acknowledgments, or trivial exchanges

When updating:
- merge new learnings into the existing file
- deduplicate similar bullets
- keep section headers: Preferences, Behaviors, Learnings, Environment, Self
- put mistakes, failed approaches, and their fixes in Learnings — phrase as actionable rules ("Avoid X; use Y instead")
- use Self for applied source customizations, pending restarts, or repo-specific paths
- preserve still-relevant existing bullets
- if nothing new is durable, return {"update": false}"""

_SKIP_PATTERNS = (
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|yep|nope|sure|done|stop|logout)\.?$",
    r"^(good|great|perfect|nice)\.?$",
)


_SKIP_AUTONOMOUS_SOURCES = frozenset({"job_followup", "continuation", "cron"})


def _should_skip_update(
    user_message: str,
    assistant_reply: str,
    settings: Settings,
    mistake_log: Optional[TurnMistakeLog] = None,
    task_source: Optional[str] = None,
) -> Optional[str]:
    if not settings.soul_enabled or not settings.soul_auto_update:
        return "disabled"
    if settings.soul_auto_update_skip_autonomous and task_source in _SKIP_AUTONOMOUS_SOURCES:
        return "autonomous_source"
    user = user_message.strip()
    assistant = assistant_reply.strip()
    if not user or not assistant:
        return "empty_turn"
    if len(user) + len(assistant) < 40:
        return "too_short"
    lowered = user.lower()
    for pattern in _SKIP_PATTERNS:
        if re.match(pattern, lowered, re.IGNORECASE):
            return "trivial_message"
    if user.lower().startswith("[autonomous"):
        return "autonomous_message"
    return None


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _build_user_prompt(
    current_soul: str,
    user_message: str,
    assistant_reply: str,
    max_file_chars: int,
    mistake_log: Optional[TurnMistakeLog] = None,
) -> str:
    soul_excerpt = current_soul
    if len(soul_excerpt) > max_file_chars:
        soul_excerpt = soul_excerpt[: max_file_chars - 20].rstrip() + "\n[... truncated ...]"
    mistakes_block = ""
    if mistake_log and mistake_log.has_entries():
        mistakes_block = (
            "\n\nFailures during this turn (extract durable lessons — what went wrong and what fixed it):\n"
            f"{mistake_log.format_for_soul()}\n"
        )
    return (
        f"Current soul.md:\n{soul_excerpt}\n\n"
        f"User message:\n{user_message.strip()[:4000]}\n\n"
        f"Assistant reply:\n{assistant_reply.strip()[:6000]}"
        f"{mistakes_block}\n\n"
        "Return merged soul.md JSON if anything durable should be saved; otherwise {\"update\": false}."
    )


async def maybe_update_soul(
    soul: SoulStore,
    llm: LLMClient,
    settings: Settings,
    bus: EventBus,
    user_message: str,
    assistant_reply: str,
    client_id: Optional[str] = None,
    mistake_log: Optional[TurnMistakeLog] = None,
    task_source: Optional[str] = None,
) -> None:
    skip_reason = _should_skip_update(
        user_message,
        assistant_reply,
        settings,
        mistake_log,
        task_source=task_source,
    )
    if skip_reason:
        await bus.publish(
            {
                "type": "soul_update_skipped",
                "reason": skip_reason,
                "client_id": client_id,
            }
        )
        return

    current = soul.load()
    messages = [
        {"role": "system", "content": SOUL_UPDATE_SYSTEM},
        {
            "role": "user",
            "content": _build_user_prompt(
                current,
                user_message,
                assistant_reply,
                settings.soul_max_file_chars,
                mistake_log=mistake_log,
            ),
        },
    ]

    try:
        loop = asyncio.get_running_loop()
        raw = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(llm.chat, messages=messages, temperature=0.1, source="soul_update"),
            ),
            timeout=max(10, min(60, settings.llm_timeout_seconds)),
        )
    except Exception as exc:
        logger.warning("soul update failed: %s", exc)
        await bus.publish(
            {
                "type": "soul_update_skipped",
                "reason": str(exc),
                "client_id": client_id,
            }
        )
        return

    payload = _extract_json(raw)
    if not payload or not payload.get("update"):
        await bus.publish(
            {
                "type": "soul_update_skipped",
                "reason": "no_update",
                "client_id": client_id,
            }
        )
        return

    content = str(payload.get("content") or "").strip()
    if not content or content == current.strip():
        await bus.publish(
            {
                "type": "soul_update_skipped",
                "reason": "empty_content",
                "client_id": client_id,
            }
        )
        return

    if len(content) > settings.soul_max_file_chars:
        content = content[: settings.soul_max_file_chars].rstrip() + "\n"

    soul.save(content)
    await bus.publish(
        {
            "type": "soul_updated",
            "chars": len(content),
            "client_id": client_id,
        }
    )
