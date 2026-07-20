from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


VALID_PLAN_ROLES = frozenset({"explore", "execute", "verify"})


@dataclass(frozen=True)
class PlanStep:
    id: str
    role: str
    goal: str
    depends_on: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    parallel_safe: bool = True
    success_criteria: List[str] = field(default_factory=list)

    def to_dict(self, *, include_status: Optional[str] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "goal": self.goal,
            "depends_on": list(self.depends_on),
            "resources": list(self.resources),
            "parallel_safe": self.parallel_safe,
            "success_criteria": list(self.success_criteria),
        }
        if include_status:
            data["status"] = include_status
        return data

    def as_subtask(self) -> Dict[str, str]:
        return {"id": self.id, "role": self.role, "goal": self.goal}

    @property
    def needs_writer_lane(self) -> bool:
        return self.role == "execute" and (not self.resources or not self.parallel_safe)


@dataclass(frozen=True)
class Plan:
    summary: str
    steps: List[PlanStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "steps": [step.to_dict(include_status="pending") for step in self.steps],
        }

    def subtasks(self) -> List[Dict[str, str]]:
        return [step.as_subtask() for step in self.steps]

    def batches(self, max_parallel: int) -> List[List[PlanStep]]:
        """Return dependency-ready execution batches with conservative resource gating."""
        limit = max(1, max_parallel)
        pending: Dict[str, PlanStep] = {step.id: step for step in self.steps}
        completed: Set[str] = set()
        batches: List[List[PlanStep]] = []

        while pending:
            ready = [
                step
                for step in pending.values()
                if all(dep in completed for dep in step.depends_on)
            ]
            if not ready:
                # Cycles are removed during normalization; this keeps runtime resilient.
                ready = [next(iter(pending.values()))]

            batch: List[PlanStep] = []
            used_resources: Set[str] = set()
            writer_lane_used = False
            for step in ready:
                if len(batch) >= limit:
                    break
                resource_set = set(step.resources)
                if step.needs_writer_lane:
                    if batch:
                        continue
                    batch.append(step)
                    writer_lane_used = True
                    break
                if writer_lane_used:
                    continue
                if resource_set and used_resources.intersection(resource_set):
                    continue
                if step.role == "execute" and not resource_set:
                    if batch:
                        continue
                    batch.append(step)
                    writer_lane_used = True
                    break
                batch.append(step)
                used_resources.update(resource_set)

            if not batch:
                batch = [ready[0]]
            for step in batch:
                pending.pop(step.id, None)
                completed.add(step.id)
            batches.append(batch)

        return batches


def _slug(value: str, fallback: str) -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    text = "".join(chars).strip("_")
    return text[:48] or fallback


def _string_list(raw: Any, *, limit: int = 8) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()][:limit]


def _dedupe_id(raw: str, used: Set[str], fallback: str) -> str:
    base = _slug(raw, fallback)
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _legacy_steps(subtasks: Sequence[Dict[str, Any]], success_criteria: Sequence[str]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    previous_id = ""
    for index, item in enumerate(subtasks):
        role = str(item.get("role") or "execute").lower()
        goal = str(item.get("goal") or "").strip()
        if role not in VALID_PLAN_ROLES or not goal:
            continue
        step_id = str(item.get("id") or f"{role}_{index + 1}")
        step: Dict[str, Any] = {
            "id": step_id,
            "role": role,
            "goal": goal,
            "depends_on": [previous_id] if previous_id else [],
            "resources": _string_list(item.get("resources"), limit=6),
            "parallel_safe": role != "execute",
            "success_criteria": list(success_criteria) if role == "verify" else [],
        }
        steps.append(step)
        previous_id = step_id
    return steps


def normalize_plan(
    payload: Dict[str, Any],
    *,
    intent: str,
    expanded_query: str,
    subtasks: Sequence[Dict[str, Any]],
    success_criteria: Sequence[str],
    max_steps: int,
) -> Plan:
    raw_plan = payload.get("plan")
    raw_steps: Iterable[Any] = []
    summary = intent
    if isinstance(raw_plan, dict):
        summary = str(raw_plan.get("summary") or intent).strip() or intent
        candidate = raw_plan.get("steps")
        if isinstance(candidate, list):
            raw_steps = candidate
    elif isinstance(payload.get("steps"), list):
        raw_steps = payload.get("steps") or []

    steps_raw = list(raw_steps)
    if not steps_raw:
        steps_raw = _legacy_steps(subtasks, success_criteria)
    if not steps_raw and expanded_query:
        steps_raw = [
            {
                "id": "execute_request",
                "role": "execute",
                "goal": expanded_query,
                "depends_on": [],
                "parallel_safe": False,
                "success_criteria": list(success_criteria),
            }
        ]

    used_ids: Set[str] = set()
    normalized: List[PlanStep] = []
    raw_to_id: Dict[str, str] = {}
    for index, item in enumerate(steps_raw[: max(1, max_steps)]):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "execute").lower()
        goal = str(item.get("goal") or item.get("description") or "").strip()
        if role not in VALID_PLAN_ROLES or not goal:
            continue
        raw_id = str(item.get("id") or f"{role}_{index + 1}")
        step_id = _dedupe_id(raw_id, used_ids, f"{role}_{index + 1}")
        raw_to_id[raw_id] = step_id
        normalized.append(
            PlanStep(
                id=step_id,
                role=role,
                goal=goal,
                depends_on=_string_list(item.get("depends_on") or item.get("dependencies"), limit=8),
                resources=_string_list(item.get("resources"), limit=8),
                parallel_safe=bool(item.get("parallel_safe", role != "execute")),
                success_criteria=_string_list(item.get("success_criteria"), limit=8),
            )
        )

    valid_ids = {step.id for step in normalized}
    rewritten: List[PlanStep] = []
    for step in normalized:
        deps: List[str] = []
        for dep in step.depends_on:
            mapped = raw_to_id.get(dep, dep)
            if mapped in valid_ids and mapped != step.id and mapped not in deps:
                deps.append(mapped)
        rewritten.append(
            PlanStep(
                id=step.id,
                role=step.role,
                goal=step.goal,
                depends_on=deps,
                resources=step.resources,
                parallel_safe=step.parallel_safe,
                success_criteria=step.success_criteria,
            )
        )

    return Plan(summary=summary, steps=_break_dependency_cycles(rewritten))


def _break_dependency_cycles(steps: Sequence[PlanStep]) -> List[PlanStep]:
    ordered: List[PlanStep] = []
    seen: Set[str] = set()
    for step in steps:
        deps = [dep for dep in step.depends_on if dep in seen]
        ordered.append(
            PlanStep(
                id=step.id,
                role=step.role,
                goal=step.goal,
                depends_on=deps,
                resources=step.resources,
                parallel_safe=step.parallel_safe,
                success_criteria=step.success_criteria,
            )
        )
        seen.add(step.id)
    return ordered
