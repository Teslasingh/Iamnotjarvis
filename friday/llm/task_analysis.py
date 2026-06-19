from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import partial
from typing import Any, Dict, List, Optional

from friday.agent.execution_intent import implies_host_execution
from friday.agent.plan import normalize_plan
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
  "subtasks": [{"role": "explore|execute|verify", "goal": "..."}],
  "plan": {
    "summary": "<mission summary>",
    "steps": [
      {
        "id": "stable_step_id",
        "role": "explore|execute|verify",
        "goal": "<specific step outcome>",
        "depends_on": ["prior_step_id"],
        "resources": ["path/module/service touched, if known"],
        "parallel_safe": true,
        "success_criteria": ["<step-specific check>", ...]
      }
    ]
  }
}

Rules:
- When the user asks about live system state (processes, tmux, docker, services, "what's running"), expanded_query must instruct run_shell on THIS host and report stdout/stderr — never rewrite into a tutorial or cheat sheet.
- When the user says "run X", "on my system", or names a shell command, preserve execution intent; do not rewrite into "explain how to".
- Never take the user prompt at face value for engineering tasks — but do NOT convert execution/status checks into documentation requests.
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
- subtasks: compatibility summary of the plan; max configured entries when orchestrate is true. Roles:
  explore = Code Analyzer (read/search/map, no mutations);
  execute = Implementation Agent (edit/run/mutate);
  verify = QA Tester (validate, self-heal, confirm success_criteria).
  Omit subtasks or use [] when orchestrate is false.
- plan: provide a dependency graph for orchestrated work. Use stable snake_case IDs. Steps with no dependency may run in parallel.
  Mark execute/write-heavy steps as parallel_safe=false unless resources are explicit and non-overlapping.
  Include resources for files/modules/services each step expects to touch. Verify steps should depend on relevant execute steps.
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
    if lowered.startswith("[autonomous"):
        return True
    if implies_host_execution(text):
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
        "plan": {"summary": original, "steps": []},
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

    plan_raw = payload.get("plan")
    plan_has_steps = (
        isinstance(plan_raw, dict)
        and isinstance(plan_raw.get("steps"), list)
        and bool(plan_raw.get("steps"))
    ) or (isinstance(payload.get("steps"), list) and bool(payload.get("steps")))

    if orchestrate and not subtasks and not plan_has_steps and complexity == "complex":
        subtasks = [
            {"role": "explore", "goal": f"Investigate context and constraints for: {intent}"},
            {"role": "execute", "goal": f"Implement the user request: {expanded}"},
            {"role": "verify", "goal": "Validate results, fix failures, confirm success criteria"},
        ]
        subtasks = subtasks[: settings.multi_agent_max_subagents]
    elif orchestrate and not subtasks and not plan_has_steps:
        orchestrate = False

    if implies_host_execution(original):
        expanded = original
        orchestrate = False
        complexity = "simple"
        subtasks = []

    plan = normalize_plan(
        payload,
        intent=intent,
        expanded_query=expanded,
        subtasks=subtasks,
        success_criteria=success_criteria,
        max_steps=settings.multi_agent_max_plan_steps,
    )
    if not orchestrate:
        plan = normalize_plan(
            {"plan": {"summary": intent, "steps": []}},
            intent=intent,
            expanded_query="",
            subtasks=[],
            success_criteria=[],
            max_steps=1,
        )
    elif not plan.steps:
        orchestrate = False
    else:
        subtasks = plan.subtasks()

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
        "plan": plan.to_dict(),
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
    complexity = str(analysis.get("complexity") or "simple").lower()
    if complexity == "simple":
        return (
            "Task analysis (internal; do not recite):\n"
            f"Intent: {analysis.get('intent', '')}\n"
            f"Request: {analysis.get('expanded_query', '')}"
        )
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
    plan = analysis.get("plan") or {}
    steps = plan.get("steps") if isinstance(plan, dict) else []
    if isinstance(steps, list) and steps:
        lines.append("Execution plan:")
        for step in steps:
            if not isinstance(step, dict):
                continue
            deps = step.get("depends_on") or []
            dep_text = f" after {', '.join(deps)}" if deps else ""
            lines.append(
                f"- {step.get('id', 'step')} ({step.get('role', 'execute')}{dep_text}): "
                f"{step.get('goal', '')}"
            )
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
                partial(llm.chat, messages=messages, temperature=0.2, source="task_analysis"),
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
            "success_criteria": result["success_criteria"][:8],
            "plan": result.get("plan") or {"summary": result["intent"], "steps": []},
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
