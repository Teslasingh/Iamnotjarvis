from __future__ import annotations

import re
import time
from pathlib import Path

from friday.agent.soul import SoulStore, bullet_already_exists, normalize_bullet_text
from friday.config import Settings
from friday.runtime.persist import atomic_write_text

USER_TEMPLATE = """# User memory

Standing preferences and instructions for Friday.

## Preferences

"""

MEMORY_TEMPLATE = """# Agent memory

Durable learnings and environment notes.

## Learnings

## Environment

"""


class PersistentMemoryStore:
    def __init__(self, workdir: Path) -> None:
        self.user_path = (workdir / "USER.md").resolve()
        self.memory_path = (workdir / "MEMORY.md").resolve()

    def _ensure(self, path: Path, template: str) -> str:
        if not path.is_file():
            atomic_write_text(path, template)
            return template
        return path.read_text(encoding="utf-8")

    def load_user(self) -> str:
        return self._ensure(self.user_path, USER_TEMPLATE)

    def load_memory(self) -> str:
        return self._ensure(self.memory_path, MEMORY_TEMPLATE)

    def append_user_bullet(self, section: str, text: str) -> bool:
        return self._append_bullet(self.user_path, USER_TEMPLATE, section, text)

    def append_memory_bullet(self, section: str, text: str) -> bool:
        return self._append_bullet(self.memory_path, MEMORY_TEMPLATE, section, text)

    def _append_bullet(self, path: Path, template: str, section: str, text: str) -> bool:
        bullet_text = text.strip()
        if not bullet_text:
            return False
        section_name = section.strip().title()
        if section_name not in {"Preferences", "Learnings", "Environment"}:
            section_name = "Learnings"
        content = self._ensure(path, template)
        existing = self._extract_section(content, section_name)
        if existing and bullet_already_exists(existing, bullet_text):
            return False
        ts = time.strftime("%Y-%m-%d", time.gmtime())
        bullet = f"- ({ts}) {bullet_text}"
        header = f"## {section_name}"
        if header not in content:
            content = content.rstrip() + f"\n\n{header}\n\n{bullet}\n"
        else:
            parts = content.split(header, 1)
            after = parts[1]
            next_h = re.search(r"\n##\s+\w+", after)
            if next_h:
                insert_at = next_h.start()
                after = after[:insert_at] + f"\n{bullet}" + after[insert_at:]
            else:
                after = after.rstrip() + f"\n{bullet}\n"
            content = parts[0] + header + after
        if section_name == "Environment":
            section_body = self._extract_section(content, section_name)
            bullets = [line for line in section_body.splitlines() if line.strip().startswith("-")]
            if len(bullets) > 5:
                trimmed = "\n".join(bullets[-5:])
                content = re.sub(
                    rf"(## Environment\s*\n)(.*?)(?=\n##\s+\w+|\Z)",
                    rf"\1\n{trimmed}\n",
                    content,
                    flags=re.DOTALL,
                )
        atomic_write_text(path, content)
        return True

    def reset_user(self) -> None:
        atomic_write_text(self.user_path, USER_TEMPLATE)

    def reset_memory(self) -> None:
        atomic_write_text(self.memory_path, MEMORY_TEMPLATE)

    def build_user_context(self, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        content = self.load_user().strip()
        if len(content) < 80:
            return ""
        if len(content) <= max_chars:
            return content
        return content[: max_chars - 20] + "\n[... truncated ...]"

    def build_memory_context(self, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        content = self.load_memory().strip()
        if len(content) < 80:
            return ""
        learnings = self._extract_section(content, "Learnings")
        prefix = ""
        if learnings:
            prefix = (
                "Agent memory — past mistakes and lessons (avoid repeating):\n"
                f"{learnings}\n\n---\n\n"
            )
        budget = max_chars - len(prefix) if prefix else max_chars
        if len(content) <= budget:
            return prefix + content
        truncated = content[: budget - 20].rstrip() + "\n[... truncated ...]"
        return prefix + truncated

    @staticmethod
    def _extract_section(content: str, section: str) -> str:
        header = f"## {section}"
        if header not in content:
            return ""
        after = content.split(header, 1)[1]
        next_header = re.search(r"\n##\s+\w+", after)
        body = after[: next_header.start()] if next_header else after
        return body.strip()


def build_combined_memory_context(
    settings: Settings,
    soul: SoulStore,
    persistent: PersistentMemoryStore,
) -> str:
    parts: list[str] = []
    budget = settings.soul_max_context_chars
    if settings.user_memory_enabled:
        user_ctx = persistent.build_user_context(min(4000, budget // 3))
        if user_ctx:
            parts.append(f"USER.md:\n{user_ctx}")
    if settings.agent_memory_enabled:
        mem_ctx = persistent.build_memory_context(min(4000, budget // 3))
        if mem_ctx:
            parts.append(mem_ctx)
    if settings.soul_enabled:
        soul_ctx = soul.build_context(budget)
        if soul_ctx:
            parts.append(soul_ctx)
    return "\n\n---\n\n".join(parts)


def mirror_remember_soul(section: str, text: str, persistent: PersistentMemoryStore) -> None:
    key = section.strip().lower()
    if key in {"preferences", "preference"}:
        persistent.append_user_bullet("Preferences", text)
    elif key in {"learnings", "learning"}:
        persistent.append_memory_bullet("Learnings", text)
    elif key in {"environment", "env"}:
        persistent.append_memory_bullet("Environment", text)
