from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

CONTEXT_FILENAMES: Tuple[str, ...] = (
    ".hermes.md",
    "AGENTS.md",
    "CLAUDE.md",
    "FRIDAY.md",
    ".cursorrules",
)


def load_context_files(workdir: Path, max_chars: int, per_file_max: int = 8000) -> str:
    if max_chars <= 0:
        return ""
    soul_lower = (workdir / "soul.md").resolve()
    soul_upper = (workdir / "SOUL.md").resolve()
    blocks: List[str] = []
    used = 0
    candidates = list(CONTEXT_FILENAMES)
    if soul_upper.is_file() and soul_upper.resolve() != soul_lower:
        candidates = list(candidates) + ["SOUL.md"]
    for name in candidates:
        path = (workdir / name).resolve()
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        if len(text) > per_file_max:
            text = text[: per_file_max - 20] + "\n[... truncated ...]"
        block = f"### Project context: {name}\n\n{text}"
        if used + len(block) > max_chars:
            remain = max_chars - used - 40
            if remain < 200:
                break
            block = block[:remain] + "\n[... truncated ...]"
        blocks.append(block)
        used += len(block)
        if used >= max_chars:
            break
    if not blocks:
        return ""
    return "Project context files:\n\n" + "\n\n---\n\n".join(blocks)
