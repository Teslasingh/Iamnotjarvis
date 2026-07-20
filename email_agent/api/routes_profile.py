"""Profile routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import storage
from errors import AppError
from profile_store import load_persisted_profile, persist_profile_resume_snapshot, resume_path, save_profile, save_resume

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["profile"])


@router.get("/profile")
def get_profile() -> dict[str, Any]:
    profile = load_persisted_profile()
    current_resume = resume_path(profile)
    profile["resume_uploaded"] = current_resume is not None
    profile["resume_file_name"] = current_resume.name if current_resume else ""
    profile["resume_profile_db_id"] = (profile.get("parsed_profile") or {}).get("db_id") or (
        storage.latest_resume_profile() or {}
    ).get("id")
    return profile


@router.get("/resume-profile")
def get_resume_profile() -> dict[str, Any]:
    profile = storage.latest_resume_profile()
    if not profile:
        raise HTTPException(404, "No parsed resume profile has been saved yet")
    return profile


@router.post("/profile")
async def update_profile(
    full_name: str = Form(""),
    email: str = Form(""),
    phone: str = Form(""),
    location: str = Form(""),
    linkedin: str = Form(""),
    portfolio: str = Form(""),
    summary: str = Form(""),
    work_experience: str = Form(""),
    skills: str = Form(""),
    job_preferences: str = Form(""),
    common_application_details: str = Form(""),
    resume_notes: str = Form(""),
    resume: UploadFile | None = File(None),
) -> dict[str, Any]:
    try:
        profile = save_profile(
            {
                "full_name": full_name,
                "email": email,
                "phone": phone,
                "location": location,
                "linkedin": linkedin,
                "portfolio": portfolio,
                "summary": summary,
                "work_experience": work_experience,
                "skills": skills,
                "job_preferences": job_preferences,
                "common_application_details": common_application_details,
                "resume_notes": resume_notes,
            }
        )
        if resume and resume.filename:
            profile = await save_resume(resume)
        else:
            profile = persist_profile_resume_snapshot(profile)
    except AppError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc
    except Exception as exc:
        logger.exception("Profile update failed")
        raise HTTPException(500, f"Profile update failed: {exc}") from exc

    current_resume = resume_path(profile)
    profile["resume_uploaded"] = current_resume is not None
    profile["resume_file_name"] = current_resume.name if current_resume else ""
    profile["resume_profile_db_id"] = (profile.get("parsed_profile") or {}).get("db_id")
    return profile
