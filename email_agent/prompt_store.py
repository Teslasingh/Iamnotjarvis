from __future__ import annotations

import json
from typing import Any

from config import DATA_DIR, ensure_directories


AGENT_PROMPTS_FILE = DATA_DIR / "agent_prompts.json"

DEFAULT_AGENT_PROMPT = """You are a resume-aware mail assistant focused on job search by default.
For each email:
1. Decide if it is job-search related (recruiter outreach, interview, application, offer, rejection).
2. If job-related, compare requirements against the candidate profile and full resume text.
3. Return match_score 0-100 as confidence in fit (100 = excellent resume match).
4. Extract required/matched/missing skills, company, role, and a short summary.
5. If not job-related, set is_job_email=false and match_score=0.
6. When drafting replies, write concise professional responses in the candidate's voice using only known profile facts."""


def load_agent_prompt() -> str:
    ensure_directories()
    if not AGENT_PROMPTS_FILE.exists():
        return DEFAULT_AGENT_PROMPT
    try:
        data = json.loads(AGENT_PROMPTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_AGENT_PROMPT
    prompt = str(data.get("agent_prompt") or "").strip()
    return prompt or DEFAULT_AGENT_PROMPT


def save_agent_prompt(text: str) -> dict[str, Any]:
    ensure_directories()
    prompt = text.strip()
    if len(prompt) < 20:
        raise ValueError("Agent prompt must be at least 20 characters.")
    payload = {"agent_prompt": prompt}
    AGENT_PROMPTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return get_agent_prompt_state()


def reset_agent_prompt() -> dict[str, Any]:
    ensure_directories()
    payload = {"agent_prompt": DEFAULT_AGENT_PROMPT}
    AGENT_PROMPTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return get_agent_prompt_state()


def get_agent_prompt_state() -> dict[str, Any]:
    prompt = load_agent_prompt()
    return {
        "agent_prompt": prompt,
        "is_default": prompt.strip() == DEFAULT_AGENT_PROMPT.strip(),
    }
