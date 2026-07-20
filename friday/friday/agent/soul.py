from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Optional

from friday.runtime.persist import atomic_write_text

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
_ENVIRONMENT_MAX_BULLETS = 5


def normalize_bullet_text(text: str) -> str:
    cleaned = re.sub(r"\(\d{4}-\d{2}-\d{2}\)", "", text.lower())
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return " ".join(cleaned.split())


def bullet_already_exists(section_body: str, text: str) -> bool:
    target = normalize_bullet_text(text)
    if not target:
        return True
    for line in section_body.splitlines():
        if not line.strip().startswith("-"):
            continue
        existing = normalize_bullet_text(line.lstrip("- ").strip())
        if not existing:
            continue
        if target == existing or target in existing or existing in target:
            return True
    return False


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
        atomic_write_text(self.path, content.rstrip() + "\n")

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

    def extract_section_body(self, section: str) -> str:
        section_name = SECTION_ALIASES.get(section.strip().lower(), section)
        header = f"## {section_name}"
        content = self.load()
        if header not in content:
            return ""
        after = content.split(header, 1)[1]
        next_header = re.search(r"\n##\s+\w+", after)
        body = after[: next_header.start()] if next_header else after
        body = body.strip()
        if not body or not _BULLET_RE.search(body):
            return ""
        return body

    def build_context(self, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        content = self.load()
        if self.is_empty(content):
            return ""
        learnings = self.extract_section_body("learnings")
        prefix = ""
        if learnings:
            prefix = (
                "Past mistakes and lessons (avoid repeating these errors):\n"
                f"{learnings}\n\n---\n\n"
            )
        budget = max_chars - len(prefix) if prefix else max_chars
        if budget < 400 and prefix:
            return prefix[:max_chars]
        if len(content) <= budget:
            return prefix + content
        truncated = content[: budget - 20].rstrip() + "\n\n[... truncated ...]"
        return prefix + truncated

    def append_bullet(self, section: str, text: str) -> bool:
        bullet_text = text.strip()
        if not bullet_text:
            return False
        section_name = SECTION_ALIASES.get(section.strip().lower(), "Learnings")
        content = self.load()
        existing_body = self.extract_section_body(section_name.lower())
        if existing_body and bullet_already_exists(existing_body, bullet_text):
            return False
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
        if section_name == "Environment":
            bullets = [line for line in section_body.splitlines() if line.strip().startswith("-")]
            if len(bullets) > _ENVIRONMENT_MAX_BULLETS:
                keep = bullets[-_ENVIRONMENT_MAX_BULLETS:]
                section_body = "\n" + "\n".join(keep) + "\n"
        self.save(before + header + section_body + rest)
        return True
