from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from friday.agent.checkpoints import CheckpointManager
from friday.agent.context_files import load_context_files
from friday.agent.persistent_memory import PersistentMemoryStore
from friday.agent.skills import SkillEntry, build_skills_catalog, discover_skills
from friday.config import Settings


@dataclass
class AgentExtras:
    context_files_context: str = ""
    skills_catalog: str = ""
    skills: Optional[Dict[str, SkillEntry]] = None
    checkpoint_manager: Optional[CheckpointManager] = None
    persistent_memory: Optional[PersistentMemoryStore] = None
    hook_runner: Any = None
    delegate_runner: Any = None
    code_exec_runner: Any = None

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "context_files_context": self.context_files_context,
            "skills_catalog": self.skills_catalog,
            "skills": self.skills,
            "checkpoint_manager": self.checkpoint_manager,
            "persistent_memory": self.persistent_memory,
            "hook_runner": self.hook_runner,
            "delegate_runner": self.delegate_runner,
            "code_exec_runner": self.code_exec_runner,
        }


def build_agent_extras(
    settings: Settings,
    workdir: Path,
    *,
    checkpoint_manager: Optional[CheckpointManager],
    persistent_memory: Optional[PersistentMemoryStore],
    hook_runner: Any,
    delegate_runner: Optional[Callable[..., Any]],
    code_exec_runner: Optional[Callable[..., Any]],
) -> AgentExtras:
    ctx_files = ""
    if settings.context_files_enabled:
        ctx_files = load_context_files(workdir, settings.context_files_max_chars)
    extra_dirs = []
    if settings.friday_skills_dirs:
        for part in settings.friday_skills_dirs.split(";"):
            p = part.strip()
            if p:
                extra_dirs.append(Path(p))
    skills = discover_skills(workdir, extra_dirs) if settings.skills_enabled else {}
    catalog = build_skills_catalog(skills, settings.skills_max_catalog_chars) if skills else ""
    return AgentExtras(
        context_files_context=ctx_files,
        skills_catalog=catalog,
        skills=skills or None,
        checkpoint_manager=checkpoint_manager,
        persistent_memory=persistent_memory,
        hook_runner=hook_runner,
        delegate_runner=delegate_runner,
        code_exec_runner=code_exec_runner,
    )
