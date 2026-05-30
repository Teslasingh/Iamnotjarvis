from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Optional

SOUL_TEMPLATE = """# Soul

Persistent learnings Friday keeps across Iamnotjarvis sessions — orchestrator behaviors, repo conventions, and durable user preferences. Not every chat is saved.

## Preferences

## Behaviors

## Learnings

## Environment

## Self
"""

SECTION_ALIASES: Dict[str, str] = {
    "preferences": "Preferences",
    "preference": "Preferences",
    "behaviors": "Behaviors",
    "behavior": "Behaviors",
    "behaviour": "Behaviors",
    "behaviours": "Behaviors",
    "learnings": "Learnings",
    "learning": "Learnings",
    "environment": "Environment",
    "env": "Environment",
    "self": "Self",
}

_SECTION_HEADER_RE = re.compile(r"^##\s+(\w+)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*-\s+", re.MULTILINE)


class SoulStore:
    """Persistent long-term memory stored as soul.md on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def load(self) -> str:
        if not self.path.is_file():
            self.save(SOUL_TEMPLATE)
            return SOUL_TEMPLATE
        return self.path.read_text(encoding="utf-8")

    def save(self, content: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(content.rstrip() + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def reset(self) -> None:
        self.save(SOUL_TEMPLATE)

    def is_empty(self, content: Optional[str] = None) -> bool:
        text = content if content is not None else self.load()
        stripped = _SECTION_HEADER_RE.sub("", text)
        stripped = re.sub(r"^#\s+Soul\s*$", "", stripped, flags=re.MULTILINE)
        stripped = re.sub(
            r"^Persistent learnings Friday keeps across sessions\..*$",
            "",
            stripped,
            flags=re.MULTILINE,
        )
        stripped = _BULLET_RE.sub("", stripped)
        return not stripped.strip()

    def build_context(self, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        content = self.load()
        if self.is_empty(content):
            return ""
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 20].rstrip() + "\n\n[... truncated ...]"

    def append_bullet(self, section: str, text: str) -> bool:
        bullet_text = text.strip()
        if not bullet_text:
            return False
        section_name = SECTION_ALIASES.get(section.strip().lower(), "Learnings")
        content = self.load()
        ts = time.strftime("%Y-%m-%d", time.gmtime())
        bullet = f"- ({ts}) {bullet_text}"
        header = f"## {section_name}"

        if header not in content:
            content = content.rstrip() + f"\n\n{header}\n\n{bullet}\n"
            self.save(content)
            return True

        parts = content.split(header, 1)
        before = parts[0]
        after = parts[1] if len(parts) > 1 else "\n"
        next_header = re.search(r"\n##\s+\w+", after)
        if next_header:
            section_body = after[: next_header.start()]
            rest = after[next_header.start() :]
        else:
            section_body = after
            rest = ""

        section_body = section_body.rstrip() + f"\n{bullet}\n"
        self.save(before + header + section_body + rest)
        return True
