"""Agent prompt routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

import prompt_store
import storage
from api.deps import AgentPromptBody

router = APIRouter(prefix="/api", tags=["prompts"])


@router.get("/agent-prompt")
def get_agent_prompt() -> dict[str, Any]:
    return prompt_store.get_agent_prompt_state()


@router.put("/agent-prompt")
def update_agent_prompt(body: AgentPromptBody) -> dict[str, Any]:
    try:
        result = prompt_store.save_agent_prompt(body.agent_prompt)
        result["pending_reanalysis"] = storage.mark_all_emails_pending_analysis()
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/agent-prompt/reset")
def reset_agent_prompt() -> dict[str, Any]:
    result = prompt_store.reset_agent_prompt()
    result["pending_reanalysis"] = storage.mark_all_emails_pending_analysis()
    return result
