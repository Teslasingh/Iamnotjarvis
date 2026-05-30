from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import partial
from typing import Any, Dict, List, Optional

from friday.config import Settings
from friday.events.bus import EventBus
from friday.llm.client import LLMClient

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM = """You analyze user requests for Friday, the Iamnotjarvis core orchestrator — a local AI agent with shell, filesystem, web search, HTTP, and SQLite tools.

Output ONLY valid JSON. No markdown fences, no commentary.

Required shape:
{
  "expanded_query": "<actionable rewritten query>",
  "intent": "<underlying goal in one sentence>",
  "edge_cases": ["<failure mode or constraint>", ...],
  "implicit_requirements": ["<security, error handling, performance, etc.>", ...],
  "success_criteria": ["<concrete pass/fail check>", ...],
  "orchestrate": false,
  "complexity": "simple|moderate|complex",
  "rationale": "<why single or multi-agent>",
  "subtasks": [{"role": "explore|execute|verify", "goal": "..."}]
}

Rules:
- Never take the user prompt at face value. Expand it with engineering judgment.
- Preserve user intent; do not answer the question or refuse the task.
- expanded_query: add missing specifics (files, formats, commands, validation steps, success criteria). Keep under 180 words.
  If already clear and specific, return the original query unchanged.
- edge_cases: infer failure modes, permissions, platform quirks, integration risks.
- implicit_requirements: infer omitted engineering constraints (security, error handling, performance, backward compatibility).
- success_criteria: concrete checks the verify role can run (import passes, test green, file exists, command exit 0).
- Use soul memory and conversation history as past experience when routing.
- Consider codebase/repo impact when the request touches friday/ source or prior session context suggests it.
- orchestrate: true ONLY when the task clearly needs multi-phase work:
  research + implement + validate, multi-module or multi-file code changes, cross-domain steps
  (web + filesystem + shell), or high ambiguity requiring separate investigation paths.
- orchestrate: false for greetings, pure Q&A, single commands, single-file edits, already-specific one-step requests.
- subtasks: max 3 entries when orchestrate is true. Roles:
  explore = Code Analyzer (read/search/map, no mutations);
  execute = Implementation Agent (edit/run/mutate);
  verify = QA Tester (validate, self-heal, confirm success_criteria).
  Omit subtasks or use [] when orchestrate is false.
- complexity: simple = one step; moderate = few related steps; complex = multi-phase or multi-domain."""

_SKIP_PATTERNS = (
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|yep|nope|sure|done|stop|logout)\.?$",
    r"^(good|great|perfect|nice)\.?$",
)

_VALID_ROLES = frozenset({"explore", "execute", "verify"})
_VALID_COMPLEXITY = frozenset({"simple", "moderate", "complex"})


def _should_skip_analysis(message: str, settings: Settings) -> bool:
    if not settings.task_analysis_enabled:
        return True
    if not settings.query_expansion_enabled:
        return True
    text = message.strip()
    if not text:
        return True
    if len(text) > settings.query_expansion_max_input_chars:
        return True
    lowered = text.lower()
    if len(text) < settings.query_expansion_min_chars:
        return True
    for pattern in _SKIP_PATTERNS:
        if re.match(pattern, lowered, re.IGNORECASE):
            return True
    return False


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


def _default_analysis(original: str) -> Dict[str, Any]:
    return {
        "expanded_query": original,
        "intent": original,
        "edge_cases": [],
        "implicit_requirements": [],
        "success_criteria": [],
        "orchestrate": False,
        "complexity": "simple",
        "rationale": "analysis skipped or unavailable",
        "subtasks": [],
    }


def _normalize_analysis(payload: Dict[str, Any], original: str, settings: Settings) -> Dict[str, Any]:
    expanded = str(payload.get("expanded_query") or original).strip() or original
    intent = str(payload.get("intent") or original).strip() or original
    edge_cases_raw = payload.get("edge_cases")
    edge_cases: List[str] = []
    if isinstance(edge_cases_raw, list):
        edge_cases = [str(item).strip() for item in edge_cases_raw if str(item).strip()][:8]

    implicit_raw = payload.get("implicit_requirements")
    implicit_requirements: List[str] = []
    if isinstance(implicit_raw, list):
        implicit_requirements = [str(item).strip() for item in implicit_raw if str(item).strip()][:8]

    success_raw = payload.get("success_criteria")
    success_criteria: List[str] = []
    if isinstance(success_raw, list):
        success_criteria = [str(item).strip() for item in success_raw if str(item).strip()][:8]

    complexity = str(payload.get("complexity") or "simple").lower()
    if complexity not in _VALID_COMPLEXITY:
        complexity = "simple"

    orchestrate = bool(payload.get("orchestrate"))
    rationale = str(payload.get("rationale") or "").strip()

    subtasks_raw = payload.get("subtasks")
    subtasks: List[Dict[str, str]] = []
    if isinstance(subtasks_raw, list):
        for item in subtasks_raw[: settings.multi_agent_max_subagents]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            goal = str(item.get("goal") or "").strip()
            if role in _VALID_ROLES and goal:
                subtasks.append({"role": role, "goal": goal})

    if orchestrate and not subtasks:
        subtasks = [
            {"role": "explore", "goal": f"Investigate context and constraints for: {intent}"},
            {"role": "execute", "goal": f"Implement the user request: {expanded}"},
            {"role": "verify", "goal": "Validate results, fix failures, confirm success criteria"},
        ]
        subtasks = subtasks[: settings.multi_agent_max_subagents]

    return {
        "original": original,
        "expanded_query": expanded,
        "intent": intent,
        "edge_cases": edge_cases,
        "implicit_requirements": implicit_requirements,
        "success_criteria": success_criteria,
        "orchestrate": orchestrate,
        "complexity": complexity,
        "rationale": rationale,
        "subtasks": subtasks,
        "applied": expanded.lower() != original.lower(),
    }


def _build_analysis_user_prompt(
    message: str,
    memory_context: str,
    soul_context: str,
    attachments: Optional[List[Dict[str, Any]]],
) -> str:
    parts = [f"User request:\n{message.strip()}"]
    if attachments:
        names = [str(a.get("name") or a.get("path") or "file") for a in attachments]
        parts.append("Attachments: " + ", ".join(names))
    if soul_context:
        parts.append("Soul memory (past experience):\n" + soul_context[:2000])
    if memory_context:
        parts.append("Recent conversation:\n" + memory_context[:1500])
    parts.append("Return analysis JSON.")
    return "\n\n".join(parts)


def build_task_brief(analysis: Dict[str, Any]) -> str:
    lines = [
        "Task analysis (use for planning; do not recite to the user):",
        f"Intent: {analysis.get('intent', '')}",
        f"Complexity: {analysis.get('complexity', 'simple')}",
        f"Expanded request: {analysis.get('expanded_query', '')}",
    ]
    implicit = analysis.get("implicit_requirements") or []
    if implicit:
        lines.append("Implicit requirements:")
        for item in implicit:
            lines.append(f"- {item}")
    edge_cases = analysis.get("edge_cases") or []
    if edge_cases:
        lines.append("Edge cases / constraints:")
        for item in edge_cases:
            lines.append(f"- {item}")
    success = analysis.get("success_criteria") or []
    if success:
        lines.append("Success criteria:")
        for item in success:
            lines.append(f"- {item}")
    rationale = analysis.get("rationale")
    if rationale:
        lines.append(f"Routing rationale: {rationale}")
    return "\n".join(lines)


async def analyze_task(
    message: str,
    llm: LLMClient,
    settings: Settings,
    bus: EventBus,
    memory_context: str = "",
    soul_context: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    original = message.strip()
    if _should_skip_analysis(original, settings):
        result = _default_analysis(original)
        result["skipped"] = True
        return result

    user_prompt = _build_analysis_user_prompt(
        original, memory_context, soul_context, attachments
    )
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]

    try:
        loop = asyncio.get_running_loop()
        raw = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(llm.chat, messages=messages, temperature=0.2),
            ),
            timeout=max(10, min(60, settings.llm_timeout_seconds)),
        )
    except Exception as exc:
        logger.warning("task analysis failed: %s", exc)
        await bus.publish(
            {
                "type": "task_analysis_skipped",
                "reason": str(exc),
                "client_id": client_id,
            }
        )
        result = _default_analysis(original)
        result["skipped"] = True
        return result

    payload = _extract_json(raw)
    if not payload:
        logger.warning("task analysis returned invalid JSON")
        await bus.publish(
            {
                "type": "task_analysis_skipped",
                "reason": "invalid_json",
                "client_id": client_id,
            }
        )
        result = _default_analysis(original)
        result["skipped"] = True
        return result

    result = _normalize_analysis(payload, original, settings)
    result["skipped"] = False

    await bus.publish(
        {
            "type": "task_analyzed",
            "complexity": result["complexity"],
            "orchestrate": result["orchestrate"],
            "intent": result["intent"][:500],
            "client_id": client_id,
        }
    )

    if result["applied"]:
        await bus.publish(
            {
                "type": "query_expanded",
                "original": original[:2000],
                "expanded": result["expanded_query"][:2000],
                "client_id": client_id,
            }
        )

    return result
