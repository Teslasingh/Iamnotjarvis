from __future__ import annotations

import unittest

from friday.agent.plan import normalize_plan
from friday.llm.task_analysis import _normalize_analysis


class _Settings:
    multi_agent_max_subagents = 3
    multi_agent_max_plan_steps = 5


class PlanNormalizationTests(unittest.TestCase):
    def test_legacy_subtasks_become_sequential_plan(self) -> None:
        plan = normalize_plan(
            {},
            intent="Improve agent",
            expanded_query="Improve the agent",
            subtasks=[
                {"role": "explore", "goal": "Map current code"},
                {"role": "execute", "goal": "Patch implementation"},
                {"role": "verify", "goal": "Run tests"},
            ],
            success_criteria=["tests pass"],
            max_steps=5,
        )

        self.assertEqual([step.id for step in plan.steps], ["explore_1", "execute_2", "verify_3"])
        self.assertEqual(plan.steps[1].depends_on, ["explore_1"])
        self.assertEqual(plan.steps[2].depends_on, ["execute_2"])
        self.assertEqual(plan.steps[2].success_criteria, ["tests pass"])

    def test_plan_steps_preserve_parallel_dependencies(self) -> None:
        plan = normalize_plan(
            {
                "plan": {
                    "summary": "Parallel work",
                    "steps": [
                        {"id": "explore_api", "role": "explore", "goal": "Inspect API"},
                        {"id": "explore_ui", "role": "explore", "goal": "Inspect UI"},
                        {
                            "id": "verify",
                            "role": "verify",
                            "goal": "Run validation",
                            "depends_on": ["explore_api", "explore_ui"],
                        },
                    ],
                }
            },
            intent="Parallel work",
            expanded_query="Parallel work",
            subtasks=[],
            success_criteria=[],
            max_steps=5,
        )

        batches = [[step.id for step in batch] for batch in plan.batches(max_parallel=3)]
        self.assertEqual(batches, [["explore_api", "explore_ui"], ["verify"]])

    def test_execute_steps_without_resources_use_writer_lane(self) -> None:
        plan = normalize_plan(
            {
                "plan": {
                    "steps": [
                        {"id": "edit_a", "role": "execute", "goal": "Edit A"},
                        {"id": "edit_b", "role": "execute", "goal": "Edit B"},
                    ]
                }
            },
            intent="Edit files",
            expanded_query="Edit files",
            subtasks=[],
            success_criteria=[],
            max_steps=5,
        )

        batches = [[step.id for step in batch] for batch in plan.batches(max_parallel=3)]
        self.assertEqual(batches, [["edit_a"], ["edit_b"]])

    def test_execute_steps_with_disjoint_resources_can_share_batch(self) -> None:
        plan = normalize_plan(
            {
                "plan": {
                    "steps": [
                        {
                            "id": "edit_api",
                            "role": "execute",
                            "goal": "Edit API",
                            "resources": ["friday/web/app.py"],
                            "parallel_safe": True,
                        },
                        {
                            "id": "edit_ui",
                            "role": "execute",
                            "goal": "Edit UI",
                            "resources": ["friday/web/static/app.js"],
                            "parallel_safe": True,
                        },
                    ]
                }
            },
            intent="Edit files",
            expanded_query="Edit files",
            subtasks=[],
            success_criteria=[],
            max_steps=5,
        )

        batches = [[step.id for step in batch] for batch in plan.batches(max_parallel=3)]
        self.assertEqual(batches, [["edit_api", "edit_ui"]])


class TaskAnalysisPlanTests(unittest.TestCase):
    def test_plan_only_payload_orchestrates_without_legacy_subtasks(self) -> None:
        result = _normalize_analysis(
            {
                "expanded_query": "Update UI and backend",
                "intent": "Update UI and backend",
                "orchestrate": True,
                "complexity": "complex",
                "plan": {
                    "summary": "Update app",
                    "steps": [
                        {"id": "inspect", "role": "explore", "goal": "Inspect code"},
                        {
                            "id": "patch",
                            "role": "execute",
                            "goal": "Patch code",
                            "depends_on": ["inspect"],
                        },
                    ],
                },
            },
            "Update UI and backend",
            _Settings(),  # type: ignore[arg-type]
        )

        self.assertTrue(result["orchestrate"])
        self.assertEqual([step["id"] for step in result["plan"]["steps"]], ["inspect", "patch"])
        self.assertEqual([step["id"] for step in result["subtasks"]], ["inspect", "patch"])


if __name__ == "__main__":
    unittest.main()
