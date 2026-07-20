from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillEntry:
    name: str
    description: str
    path: Path
    skill_dir: Path


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    block = match.group(1)
    body = text[match.end() :]
    meta: Dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip().lower()] = val.strip().strip('"').strip("'")
    return meta, body


def discover_skills(workdir: Path, extra_dirs: List[Path]) -> Dict[str, SkillEntry]:
    found: Dict[str, SkillEntry] = {}
    roots: List[Path] = [workdir / "skills"]
    roots.extend(extra_dirs)
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                text = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue
            meta, _body = _parse_frontmatter(text)
            name = (meta.get("name") or child.name).strip().lower()
            desc = (meta.get("description") or "").strip()
            if not name or not desc:
                continue
            if name in found:
                continue
            found[name] = SkillEntry(
                name=name,
                description=desc[:1024],
                path=skill_file,
                skill_dir=child,
            )
    return found


def build_skills_catalog(skills: Dict[str, SkillEntry], max_chars: int) -> str:
    if not skills or max_chars <= 0:
        return ""
    lines = [
        "Available skills (call load_skill with the skill name before following its workflow):",
    ]
    for entry in sorted(skills.values(), key=lambda e: e.name):
        lines.append(f"- {entry.name}: {entry.description}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n[... truncated ...]"
    return text


def load_skill_body(entry: SkillEntry, max_chars: int) -> str:
    text = entry.path.read_text(encoding="utf-8")
    _meta, body = _parse_frontmatter(text)
    body = body.strip()
    if len(body) > max_chars:
        return body[: max_chars - 20] + "\n[... truncated ...]"
    return body
