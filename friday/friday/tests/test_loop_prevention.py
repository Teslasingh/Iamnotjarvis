from __future__ import annotations

import unittest

from friday.agent.autonomy import (
    _job_outcome_is_clean_success,
    is_diagnostic_shell_command,
    normalize_shell_command,
)
from friday.agent.execution_intent import implies_host_execution
from friday.agent.memory import MemoryStore
from friday.agent.soul import bullet_already_exists, normalize_bullet_text
from friday.llm.soul_update import _should_skip_update


class DiagnosticCommandTests(unittest.TestCase):
    def test_tesseract_version_is_diagnostic(self) -> None:
        self.assertTrue(is_diagnostic_shell_command("tesseract --version"))
        self.assertTrue(is_diagnostic_shell_command("& \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\" --version"))

    def test_general_get_childitem_not_diagnostic(self) -> None:
        self.assertFalse(
            is_diagnostic_shell_command(
                "Get-ChildItem C:\\Users\\tesla\\Documents -Recurse -Filter *.pdf"
            )
        )

    def test_path_probe_get_childitem_is_diagnostic(self) -> None:
        self.assertTrue(
            is_diagnostic_shell_command(
                "Get-ChildItem 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
            )
        )

    def test_normalize_shell_command(self) -> None:
        self.assertEqual(
            normalize_shell_command("  Where.EXE   Tesseract  "),
            "where.exe tesseract",
        )


class ExecutionIntentTests(unittest.TestCase):
    def test_tmux_running_query(self) -> None:
        self.assertTrue(implies_host_execution("what are the task running in tmux"))

    def test_run_on_system(self) -> None:
        self.assertTrue(implies_host_execution("tmux ls run it on my system"))

    def test_pure_concept_question(self) -> None:
        self.assertFalse(implies_host_execution("what is tmux"))


class JobOutcomeTests(unittest.TestCase):
    def test_clean_success_skips_followup(self) -> None:
        ev = {
            "return_code": 0,
            "status": "done",
            "report": "Shell job abc finished OK (exit 0)",
            "stderr_tail": "",
            "outcome": {"suspect_failure": False, "exit_ok": True},
        }
        self.assertTrue(_job_outcome_is_clean_success(ev))

    def test_suspect_failure_not_clean(self) -> None:
        ev = {
            "return_code": 0,
            "status": "done",
            "report": "finished OK",
            "stderr_tail": "",
            "outcome": {"suspect_failure": True, "exit_ok": False},
        }
        self.assertFalse(_job_outcome_is_clean_success(ev))

    def test_tmux_no_server_not_suspect(self) -> None:
        from friday.runtime.shell_analysis import analyze_shell_streams

        outcome = analyze_shell_streams(
            "",
            "error connecting to /tmp/tmux-0/default (No such file or directory)\n",
            1,
        )
        self.assertFalse(outcome["suspect_failure"])


class MemoryStoreTests(unittest.TestCase):
    def test_build_context_skips_autonomous_turns(self) -> None:
        store = MemoryStore(recent_turns=10, skip_autonomous_in_context=True)
        store.append_turn("hello", "hi there", source="user")
        store.append_turn(
            "[Autonomous job follow-up] tesseract --version",
            "Checking PATH again...",
            source="job_followup",
            autonomous=True,
        )
        ctx = store.build_context()
        self.assertIn("hello", ctx)
        self.assertNotIn("tesseract", ctx.lower())


class SoulUpdateSkipTests(unittest.TestCase):
    def test_skip_autonomous_source(self) -> None:
        class _Settings:
            soul_enabled = True
            soul_auto_update = True
            soul_auto_update_skip_autonomous = True

        reason = _should_skip_update(
            "[job follow-up] tesseract",
            "Verified install.",
            _Settings(),  # type: ignore[arg-type]
            task_source="job_followup",
        )
        self.assertEqual(reason, "autonomous_source")


class BulletDedupTests(unittest.TestCase):
    def test_normalize_and_detect_duplicate(self) -> None:
        a = normalize_bullet_text("(2026-06-04) Tesseract OCR must be installed separately")
        b = normalize_bullet_text("Tesseract OCR must be installed separately on Windows")
        self.assertTrue(a in b or b in a or a == b)

    def test_bullet_already_exists(self) -> None:
        body = "- (2026-06-04) Use Test-Path on Windows instead of bare Get-ChildItem"
        self.assertTrue(
            bullet_already_exists(body, "Use Test-Path on Windows instead of bare Get-ChildItem")
        )


if __name__ == "__main__":
    unittest.main()
