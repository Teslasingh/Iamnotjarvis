"""Unit tests for user override learning and preference synthesis."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.user_learning import (
    dedup_learning_lines,
    detect_snapshot_delta,
    filter_agent_era_events,
    format_user_event,
)
from src.agent.user_preferences import (
    UserPreferenceStore,
    UserPreferenceSynthesizer,
    parse_rules_from_llm,
)


class TestFilterAgentEraEvents:
    def test_drops_agent_era_clicks_when_step_had_click_action(self):
        events = [
            {"type": "click", "text": "Submit", "ts": 100.5},
            {"type": "click", "text": "Next job", "ts": 200.0},
        ]
        filtered = filter_agent_era_events(
            events,
            {"click_element_by_index"},
            step_start=100.0,
            step_end=101.0,
        )
        assert len(filtered) == 1
        assert filtered[0]["text"] == "Next job"

    def test_keeps_clicks_outside_agent_window(self):
        events = [{"type": "click", "text": "Easy Apply", "ts": 50.0}]
        filtered = filter_agent_era_events(
            events,
            {"click_element_by_index"},
            step_start=100.0,
            step_end=110.0,
        )
        assert len(filtered) == 1

    def test_drops_agent_era_scroll_when_step_scrolled(self):
        events = [{"type": "scroll", "scrollY": 400, "ts": 105.0}]
        filtered = filter_agent_era_events(
            events,
            {"scroll_down"},
            step_start=100.0,
            step_end=110.0,
        )
        assert filtered == []


class TestFormatUserEvent:
    def test_input_field_edit(self):
        line = format_user_event(
            {
                "type": "input",
                "field": "Notice period",
                "value": "60 days",
                "url": "https://example.com/apply",
            }
        )
        assert line is not None
        assert "Notice period" in line
        assert "60 days" in line

    def test_scroll_with_container(self):
        line = format_user_event(
            {
                "type": "scroll",
                "scrollY": 420,
                "container": ".jobs-easy-apply-content",
                "url": "https://linkedin.com",
            }
        )
        assert line is not None
        assert ".jobs-easy-apply-content" in line
        assert "420" in line


class TestDetectSnapshotDelta:
    def test_container_scroll_without_window_change(self):
        previous = {
            "url": "https://linkedin.com/apply",
            "scrollY": 0,
            "containerScrollTop": 100,
        }
        current = {
            "url": "https://linkedin.com/apply",
            "scrollY": 0,
            "containerScrollTop": 250,
            "containerSource": ".modal",
        }
        lines = detect_snapshot_delta(previous, current, set())
        assert len(lines) == 1
        assert "inside .modal" in lines[0]
        assert "manually" in lines[0]

    def test_no_container_delta_when_agent_scrolled(self):
        previous = {"url": "https://x.com", "scrollY": 0, "containerScrollTop": 0}
        current = {
            "url": "https://x.com",
            "scrollY": 0,
            "containerScrollTop": 200,
            "containerSource": "DIV",
        }
        lines = detect_snapshot_delta(previous, current, {"scroll_down"})
        assert lines == []


class TestDedupLearningLines:
    def test_prefers_delta_scroll_over_js_scroll(self):
        lines = dedup_learning_lines(
            [
                "User scrolled down manually (~0px → ~200px)",
                "User scrolled to ~200px on https://linkedin.com",
            ]
        )
        assert len(lines) == 1
        assert "manually" in lines[0]


class TestUserPreferenceStore:
    def test_merge_rules_dedupes_and_caps(self, tmp_path, monkeypatch):
        monkeypatch.setenv("USER_PREFERENCE_MAX_RULES", "5")
        path = tmp_path / "prefs.json"
        store = UserPreferenceStore()
        store.merge_rules(["Skip long application forms", "Prefer Easy Apply only"])
        store.merge_rules(["Skip long application forms", "Scroll inside modals first"])
        assert len(store.preferences) == 3
        store.save(path)
        loaded = UserPreferenceStore.load(path)
        assert len(loaded.preferences) == 3

    def test_record_raw_dedupes_recent(self):
        store = UserPreferenceStore()
        store.record_raw("User clicked Easy Apply")
        store.record_raw("User clicked Easy Apply")
        assert len(store.unsynthesized_raw) == 1

    def test_format_injection(self):
        store = UserPreferenceStore()
        store.merge_rules(["Skip external apply jobs"])
        block = store.format_injection()
        assert block is not None
        assert "[User preferences" in block
        assert "Skip external apply" in block


class TestParseRulesFromLlm:
    def test_parses_bullet_lines(self):
        content = """
Here are the rules:
- Skip long application forms
- Prefer scrolling inside modals
* Use Easy Apply only
"""
        rules = parse_rules_from_llm(content)
        assert len(rules) == 3
        assert "Skip long application forms" in rules[0]


class TestUserPreferenceSynthesizer:
    def test_synthesize_merges_llm_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("USER_PREFERENCE_SYNTHESIS_MIN_EVENTS", "2")
        store = UserPreferenceStore()
        store.record_raw("User clicked Skip")
        store.record_raw("User clicked Skip again")

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "- Skip jobs when user clicks Skip\n- Follow user navigation"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        synthesizer = UserPreferenceSynthesizer(mock_llm)
        result = asyncio.run(synthesizer.synthesize(store))
        assert result is True
        assert len(store.preferences) >= 1
        assert store.unsynthesized_raw == []
