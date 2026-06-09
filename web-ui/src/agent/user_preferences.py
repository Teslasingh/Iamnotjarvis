"""
Cross-run user preference store and LLM synthesis from manual browser corrections.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_WEBUI_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PREFERENCE_FILE = _WEBUI_ROOT / "tmp" / "webui_session" / "user_preferences.json"

_SYNTHESIS_SYSTEM = """You summarize manual browser corrections into concise preference rules for an autonomous web agent.
Output 3-8 imperative rules, one per line, each starting with "- ".
Rules must be actionable (what to do/avoid), not descriptions of single events.
Generalize repeated patterns; keep site-specific detail only when observed multiple times.
Do not include markdown headers or numbering — only bullet lines."""

_SYNTHESIS_USER_TEMPLATE = """Recent manual corrections (user overrode the agent):
{raw_lines}

Existing preference rules (update/merge; drop obsolete rules):
{existing_rules}

Produce an updated set of preference rules as bullet lines only."""


def preference_synthesis_enabled() -> bool:
    return os.getenv("USER_PREFERENCE_SYNTHESIS_ENABLED", "true").lower() in (
        "1", "true", "yes", "on"
    )


def synthesis_min_events() -> int:
    try:
        return max(1, int(os.getenv("USER_PREFERENCE_SYNTHESIS_MIN_EVENTS", "3")))
    except ValueError:
        return 3


def preference_max_rules() -> int:
    try:
        return max(3, int(os.getenv("USER_PREFERENCE_MAX_RULES", "15")))
    except ValueError:
        return 15


def _preference_file_path() -> Path:
    custom = os.getenv("USER_PREFERENCE_FILE", "").strip()
    if custom:
        return Path(custom)
    return _DEFAULT_PREFERENCE_FILE


def _normalize_rule(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@dataclass
class PreferenceRule:
    rule: str
    confidence: str = "medium"
    updated_at: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "confidence": self.confidence,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceRule":
        return cls(
            rule=str(data.get("rule") or "")[:500],
            confidence=str(data.get("confidence") or "medium")[:20],
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class UserPreferenceStore:
    """Durable preference rules synthesized from user overrides."""

    preferences: List[PreferenceRule] = field(default_factory=list)
    unsynthesized_raw: List[str] = field(default_factory=list)
    updated_at: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "UserPreferenceStore":
        path = path or _preference_file_path()
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            prefs = [
                PreferenceRule.from_dict(p)
                for p in (data.get("preferences") or [])
                if isinstance(p, dict)
            ]
            raw = [str(x)[:500] for x in (data.get("unsynthesized_raw") or []) if x]
            return cls(
                preferences=prefs,
                unsynthesized_raw=raw[-30:],
                updated_at=str(data.get("updated_at") or ""),
            )
        except Exception as exc:
            logger.warning("Could not load user preferences from %s: %s", path, exc)
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        path = path or _preference_file_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.updated_at = datetime.now(timezone.utc).isoformat()
            payload = {
                "preferences": [p.to_dict() for p in self.preferences],
                "unsynthesized_raw": self.unsynthesized_raw[-30:],
                "updated_at": self.updated_at,
            }
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save user preferences: %s", exc)

    @classmethod
    def clear_persisted(cls, path: Optional[Path] = None) -> None:
        path = path or _preference_file_path()
        try:
            if path.is_file():
                path.unlink()
        except Exception as exc:
            logger.warning("Could not clear preference file: %s", exc)

    def reset(self) -> None:
        self.preferences.clear()
        self.unsynthesized_raw.clear()
        self.updated_at = ""

    def record_raw(self, message: str) -> None:
        msg = (message or "").strip()[:500]
        if not msg:
            return
        for existing in self.unsynthesized_raw[-6:]:
            if existing == msg:
                return
        self.unsynthesized_raw.append(msg)
        self.unsynthesized_raw = self.unsynthesized_raw[-30:]

    def merge_rules(self, new_rules: List[str]) -> None:
        """Merge synthesized rules; cap total and dedupe by normalized text."""
        if not new_rules:
            return
        now = datetime.now(timezone.utc).isoformat()
        existing_norm = {_normalize_rule(p.rule) for p in self.preferences}
        for rule_text in new_rules:
            text = (rule_text or "").strip()[:500]
            if not text:
                continue
            norm = _normalize_rule(text)
            if norm in existing_norm:
                continue
            self.preferences.append(
                PreferenceRule(rule=text, confidence="high", updated_at=now)
            )
            existing_norm.add(norm)
        cap = preference_max_rules()
        self.preferences = self.preferences[-cap:]

    def format_injection(self) -> Optional[str]:
        if not self.preferences:
            return None
        lines = [
            "[User preferences — learned from past manual corrections; follow unless the current task explicitly contradicts]",
        ]
        for pref in self.preferences[-preference_max_rules() :]:
            lines.append(f"- {pref.rule[:220]}")
        lines.append("- Update your `memory` field when applying a learned preference.")
        return "\n".join(lines)

    def needs_synthesis(self) -> bool:
        if not preference_synthesis_enabled():
            return False
        return len(self.unsynthesized_raw) >= synthesis_min_events()


def parse_rules_from_llm(content: str) -> List[str]:
    """Extract bullet rules from LLM response text."""
    if not content:
        return []
    rules: list[str] = []
    for line in str(content).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^[-*•]\s+(.+)$", stripped)
        if m:
            rules.append(m.group(1).strip())
        elif re.match(r"^\d+[.)]\s+(.+)$", stripped):
            rules.append(re.sub(r"^\d+[.)]\s+", "", stripped).strip())
    return [r for r in rules if len(r) > 5]


class UserPreferenceSynthesizer:
    """Summarize raw override events into durable preference rules via LLM."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self._llm = llm

    def set_llm(self, llm: Optional[BaseChatModel]) -> None:
        self._llm = llm

    async def synthesize(self, store: UserPreferenceStore) -> bool:
        """Run LLM synthesis when enough raw events are queued. Returns True if updated."""
        if not store.needs_synthesis() or self._llm is None:
            return False

        raw_lines = "\n".join(f"- {line}" for line in store.unsynthesized_raw[-15:])
        existing = "\n".join(f"- {p.rule}" for p in store.preferences[-preference_max_rules() :])
        if not existing.strip():
            existing = "(none yet)"

        prompt = _SYNTHESIS_USER_TEMPLATE.format(
            raw_lines=raw_lines,
            existing_rules=existing,
        )

        try:
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=_SYNTHESIS_SYSTEM),
                    HumanMessage(content=prompt),
                ]
            )
            content = getattr(response, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in content
                )
            rules = parse_rules_from_llm(str(content))
            if not rules:
                logger.warning("Preference synthesis returned no parseable rules")
                return False
            store.merge_rules(rules)
            store.unsynthesized_raw.clear()
            store.save()
            logger.info("Synthesized %d user preference rule(s)", len(store.preferences))
            return True
        except Exception as exc:
            logger.warning("Preference synthesis failed: %s", exc)
            return False

    async def synthesize_remaining(self, store: UserPreferenceStore) -> bool:
        """Final pass on task end — synthesize even if below min threshold."""
        if not preference_synthesis_enabled() or self._llm is None:
            return False
        if not store.unsynthesized_raw:
            return False
        if len(store.unsynthesized_raw) < synthesis_min_events():
            old_min = os.environ.get("USER_PREFERENCE_SYNTHESIS_MIN_EVENTS")
            os.environ["USER_PREFERENCE_SYNTHESIS_MIN_EVENTS"] = "1"
            try:
                return await self.synthesize(store)
            finally:
                if old_min is None:
                    os.environ.pop("USER_PREFERENCE_SYNTHESIS_MIN_EVENTS", None)
                else:
                    os.environ["USER_PREFERENCE_SYNTHESIS_MIN_EVENTS"] = old_min
        return await self.synthesize(store)
