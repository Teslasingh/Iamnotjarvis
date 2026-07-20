"""Email analysis and reply drafting (LLM + heuristic fallback)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import settings
from constants import ANALYSIS_STATUSES, JOB_KEYWORDS, RELEVANT_SCORE
from llm import as_list, chat_completion, chat_completion_json
from prompt_store import load_agent_prompt
from resume_parser import SKILL_CATALOG

logger = logging.getLogger(__name__)


def analyze_email(
    email: dict[str, Any],
    profile: dict[str, Any],
    agent_prompt: str | None = None,
) -> dict[str, Any]:
    instructions = (agent_prompt or load_agent_prompt()).strip()
    if not settings.llm_configured:
        return heuristic_analysis(email, profile, instructions)

    prompt = {
        "email": {
            "from": email.get("sender"),
            "subject": email.get("subject"),
            "snippet": email.get("snippet"),
            "body": (email.get("body_text") or "")[:12000],
        },
        "candidate_profile": _profile_for_prompt(profile),
        "agent_instructions": instructions,
    }
    try:
        analysis = chat_completion_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You analyze Gmail messages for a resume-aware mail assistant. "
                        "Return strict JSON only with keys: is_job_email, company, job_title, "
                        "job_type, match_score, confidence, required_skills, matched_skills, missing_skills, "
                        "summary, match_explanation, confidence_explanation, non_match_reason, "
                        "recommended_action, needs_reply, needs_resume. "
                        "Use arrays for skill fields. For job emails, confidence should equal match_score / 100. "
                        "confidence_explanation must explain resume-based fit. Do not invent facts."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.1,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM analysis failed; using heuristic fallback: %s", exc)
        analysis = heuristic_analysis(email, profile)
        analysis["match_explanation"] = f"LLM analysis failed; used heuristic fallback. Error: {exc}"

    return normalize_analysis(analysis, email)


def draft_reply(
    email: dict[str, Any],
    analysis: dict[str, Any],
    profile: dict[str, Any],
    agent_prompt: str | None = None,
) -> dict[str, Any]:
    attach_resume = bool(analysis.get("needs_resume"))
    subject = email.get("subject") or "Job opportunity"
    instructions = (agent_prompt or load_agent_prompt()).strip()
    if not settings.llm_configured:
        body = heuristic_reply(email, analysis, profile)
        return {"subject": subject, "body": body, "attach_resume": attach_resume}

    prompt = {
        "email": {
            "from": email.get("sender"),
            "subject": email.get("subject"),
            "body": (email.get("body_text") or "")[:12000],
        },
        "analysis": analysis,
        "candidate_profile": _profile_for_prompt(profile),
        "agent_instructions": instructions,
    }
    try:
        body = chat_completion(
            [
                {
                    "role": "system",
                    "content": "You write professional email replies for job-search communication. Return only the email body.",
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM draft failed; using heuristic fallback: %s", exc)
        body = heuristic_reply(email, analysis, profile)
        body += f"\n\n[Draft note: LLM drafting failed and a fallback draft was used: {exc}]"

    return {"subject": subject, "body": body, "attach_resume": attach_resume}


def normalize_analysis(analysis: dict[str, Any], email: dict[str, Any]) -> dict[str, Any]:
    fallback = heuristic_analysis(email, {})
    merged = fallback | {key: value for key, value in analysis.items() if value not in (None, "")}
    is_job = bool(merged.get("is_job_email", merged.get("is_job")))
    merged["is_job_email"] = is_job
    merged["is_job"] = is_job
    try:
        merged["match_score"] = max(0, min(100, int(round(float(merged.get("match_score") or 0)))))
    except (TypeError, ValueError):
        merged["match_score"] = fallback["match_score"]
    merged["confidence"] = merged["match_score"] / 100
    for key in ("required_skills", "matched_skills", "missing_skills"):
        merged[key] = as_list(merged.get(key))
    if not merged.get("status") or merged.get("status") not in ANALYSIS_STATUSES:
        merged["status"] = _status_for_analysis(is_job, merged["match_score"])
    merged["needs_reply"] = bool(merged.get("needs_reply"))
    merged["needs_resume"] = bool(merged.get("needs_resume"))
    merged["company"] = str(merged.get("company") or "")
    merged["job_title"] = str(merged.get("job_title") or merged.get("role") or _guess_role(email.get("subject", "")) or "")
    merged["role"] = merged["job_title"]
    merged["job_type"] = str(merged.get("job_type") or _guess_job_type(email) or "")
    merged["summary"] = str(merged.get("summary") or email.get("snippet") or "")
    merged["match_explanation"] = str(merged.get("match_explanation") or merged.get("reasoning") or "")
    merged["confidence_explanation"] = str(
        merged.get("confidence_explanation")
        or merged.get("match_explanation")
        or merged.get("reasoning")
        or ""
    )
    merged["non_match_reason"] = str(merged.get("non_match_reason") or "")
    merged["recommended_action"] = str(merged.get("recommended_action") or merged.get("action_needed") or "Review")
    merged["action_needed"] = merged["recommended_action"]
    return merged


def heuristic_analysis(
    email: dict[str, Any],
    profile: dict[str, Any] | None = None,
    agent_prompt: str | None = None,
) -> dict[str, Any]:
    profile = profile or {}
    text = f"{email.get('subject', '')}\n{email.get('snippet', '')}\n{email.get('body_text', '')}".lower()
    matches = [keyword for keyword in JOB_KEYWORDS if keyword in text]
    is_job = bool(matches)
    required_skills = _skills_in_text(text)
    candidate_skills = _profile_skills(profile)
    matched_skills = [skill for skill in required_skills if skill.lower() in {item.lower() for item in candidate_skills}]
    missing_skills = [skill for skill in required_skills if skill not in matched_skills]
    if is_job and required_skills:
        overlap = len(matched_skills) / max(len(required_skills), 1)
        score = int(45 + overlap * 50)
    elif is_job:
        score = min(70, 35 + len(matches) * 7)
    else:
        score = 0
    if any(word in text for word in ("unfortunately", "not moving forward", "not selected", "rejected")):
        score = min(score, 25)
    needs_resume = "resume" in text or "cv" in text
    needs_reply = is_job and not any(word in text for word in ("unfortunately", "not moving forward", "not selected"))
    status = _status_for_analysis(is_job, score)
    return {
        "is_job_email": is_job,
        "is_job": is_job,
        "confidence": score / 100,
        "status": status,
        "match_score": score,
        "company": "",
        "job_title": _guess_role(email.get("subject", "")),
        "role": _guess_role(email.get("subject", "")),
        "job_type": _guess_job_type(email),
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommended_action": "Review and reply" if needs_reply else "Review",
        "action_needed": "Review and reply" if needs_reply else "Review",
        "summary": email.get("snippet", ""),
        "needs_reply": needs_reply,
        "needs_resume": needs_resume,
        "application_details": {},
        "match_explanation": _heuristic_explanation(is_job, score, matched_skills, missing_skills),
        "confidence_explanation": _heuristic_explanation(is_job, score, matched_skills, missing_skills),
        "non_match_reason": "" if is_job else "No strong job-related keywords were found.",
        "reasoning": "Keyword-based fallback classification and skill-overlap scoring.",
    }


def heuristic_reply(email: dict[str, Any], analysis: dict[str, Any], profile: dict[str, Any]) -> str:
    name = profile.get("full_name") or "Candidate"
    role = analysis.get("role") or "the role"
    company = analysis.get("company") or "your team"
    skills = profile.get("skills") or "my relevant skills and experience"
    lines = [
        "Hello,",
        "",
        f"Thank you for reaching out about {role} with {company}. I appreciate the opportunity and would be happy to discuss how {skills} align with what you are looking for.",
    ]
    if analysis.get("needs_resume"):
        lines.append("I have attached my resume for your review.")
    lines.extend(
        [
            "Please let me know the next steps or any additional details you need from me.",
            "",
            "Best regards,",
            name,
        ]
    )
    return "\n".join(lines)


def _profile_for_prompt(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_name": profile.get("full_name"),
        "email": profile.get("email"),
        "phone": profile.get("phone"),
        "location": profile.get("location"),
        "linkedin": profile.get("linkedin"),
        "portfolio": profile.get("portfolio"),
        "summary": profile.get("summary"),
        "work_experience": profile.get("work_experience"),
        "skills": profile.get("skills"),
        "job_preferences": profile.get("job_preferences"),
        "common_application_details": profile.get("common_application_details"),
        "resume_notes": profile.get("resume_notes"),
        "parsed_profile": profile.get("parsed_profile"),
        "profile_summary": profile.get("profile_summary"),
        "resume_text": (profile.get("resume_text") or "")[:12000],
    }


def _guess_role(subject: str) -> str:
    match = re.search(r"(?:role|position|opportunity)\s*:?\s*(.+)", subject, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _status_for_analysis(is_job: bool, score: int) -> str:
    if not is_job:
        return "Not Relevant"
    if score >= RELEVANT_SCORE:
        return "Relevant"
    return "Analyzed"


def _skills_in_text(text: str) -> list[str]:
    found = [skill for skill in SKILL_CATALOG if re.search(rf"(?<![\w+#.-]){re.escape(skill)}(?![\w+#.-])", text.lower())]
    return sorted(set(found), key=found.index)


def _profile_skills(profile: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(as_list(profile.get("skills")))
    parsed = profile.get("parsed_profile") or {}
    if isinstance(parsed, dict):
        values.extend(as_list(parsed.get("skills")))
        values.extend(as_list(parsed.get("tools")))
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _guess_job_type(email: dict[str, Any]) -> str:
    text = f"{email.get('subject', '')}\n{email.get('body_text', '')}".lower()
    if "intern" in text:
        return "Internship"
    if "contract" in text or "freelance" in text:
        return "Contract"
    if "part-time" in text or "part time" in text:
        return "Part-time"
    if "remote" in text:
        return "Remote"
    if "full-time" in text or "full time" in text:
        return "Full-time"
    return ""


def _heuristic_explanation(is_job: bool, score: int, matched_skills: list[str], missing_skills: list[str]) -> str:
    if not is_job:
        return "The message does not look like a job opportunity based on keyword analysis."
    matched = ", ".join(matched_skills[:6]) or "no explicit skill matches"
    missing = ", ".join(missing_skills[:6]) or "no obvious missing skills"
    return f"Heuristic score {score}/100 based on matched skills ({matched}) and gaps ({missing})."
