from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import UploadFile

import storage
from config import PROFILE_FILE, UPLOAD_DIR, ensure_directories
from resume_parser import merge_resume_profile, parse_resume_file


DEFAULT_PROFILE: dict[str, Any] = {
    "full_name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "portfolio": "",
    "summary": "",
    "work_experience": "",
    "skills": "",
    "job_preferences": "",
    "common_application_details": "",
    "resume_path": "",
    "resume_notes": "",
    "resume_text": "",
    "parsed_profile": {},
    "profile_summary": "",
}


def load_profile() -> dict[str, Any]:
    ensure_directories()
    if not PROFILE_FILE.exists():
        return DEFAULT_PROFILE.copy()
    try:
        data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_PROFILE.copy()
    profile = DEFAULT_PROFILE.copy()
    profile.update({key: value for key, value in data.items() if key in profile})
    return profile


def save_profile(updates: dict[str, Any]) -> dict[str, Any]:
    profile = load_profile()
    for key, value in updates.items():
        if key in profile and value is not None:
            profile[key] = value
    PROFILE_FILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def write_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized = DEFAULT_PROFILE.copy()
    normalized.update({key: value for key, value in profile.items() if key in normalized})
    PROFILE_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def load_persisted_profile() -> dict[str, Any]:
    profile = load_profile()
    hydrated = hydrate_profile_from_resume_db(profile)
    if hydrated != profile:
        return write_profile(hydrated)
    return hydrated


def hydrate_profile_from_resume_db(profile: dict[str, Any]) -> dict[str, Any]:
    db_profile = _resume_db_profile(profile)
    if not db_profile:
        return profile

    hydrated = profile.copy()
    parsed_profile = db_profile.get("parsed_profile") or {}
    parsed_profile["db_id"] = db_profile.get("id")
    hydrated["parsed_profile"] = parsed_profile
    hydrated["resume_text"] = db_profile.get("resume_text") or hydrated.get("resume_text", "")
    hydrated["profile_summary"] = (
        db_profile.get("natural_language_summary")
        or parsed_profile.get("natural_language_summary")
        or hydrated.get("profile_summary", "")
    )
    hydrated["resume_path"] = hydrated.get("resume_path") or db_profile.get("resume_path") or ""

    for key in ("full_name", "email", "phone", "location", "linkedin", "portfolio", "summary", "work_experience", "job_preferences"):
        if not hydrated.get(key) and parsed_profile.get(key):
            hydrated[key] = parsed_profile[key]
    if not hydrated.get("skills") and parsed_profile.get("skills"):
        hydrated["skills"] = ", ".join(parsed_profile["skills"])
    return hydrated


def persist_profile_resume_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    resume = resume_path(profile)
    parsed_profile = profile.get("parsed_profile") or {}
    if not resume or not parsed_profile:
        return profile
    db_id = parsed_profile.get("db_id")
    existing = storage.get_resume_profile(int(db_id)) if db_id else storage.latest_resume_profile_for_path(str(resume))
    if existing:
        parsed_profile["db_id"] = existing.get("id")
        profile["parsed_profile"] = parsed_profile
        return write_profile(profile)
    parsed_data = {
        "resume_text": profile.get("resume_text", ""),
        "parsed_profile": {key: value for key, value in parsed_profile.items() if key != "db_id"},
        "profile_summary": profile.get("profile_summary", ""),
    }
    db_profile = storage.save_resume_profile(str(resume), parsed_data)
    parsed_profile["db_id"] = db_profile.get("id")
    profile["parsed_profile"] = parsed_profile
    return write_profile(profile)


def _resume_db_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    parsed_profile = profile.get("parsed_profile") or {}
    db_id = parsed_profile.get("db_id")
    if db_id:
        try:
            db_profile = storage.get_resume_profile(int(db_id))
        except (TypeError, ValueError):
            db_profile = None
        if db_profile:
            return db_profile
    current_resume = resume_path(profile)
    if current_resume:
        db_profile = storage.latest_resume_profile_for_path(str(current_resume))
        if db_profile:
            return db_profile
    return storage.latest_resume_profile()


def resume_path(profile: dict[str, Any] | None = None) -> Path | None:
    profile = profile or load_profile()
    raw_path = profile.get("resume_path")
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


async def save_resume(upload: UploadFile) -> dict[str, Any]:
    ensure_directories()
    filename = Path(upload.filename or "resume").name
    destination = UPLOAD_DIR / filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    profile = save_profile({"resume_path": str(destination)})
    try:
        parsed_data = parse_resume_file(destination, profile)
        db_profile = storage.save_resume_profile(str(destination), parsed_data)
        merged = merge_resume_profile(profile, parsed_data)
        merged["parsed_profile"] = merged.get("parsed_profile") or {}
        merged["parsed_profile"]["db_id"] = db_profile.get("id")
        return write_profile(merged)
    except Exception as exc:
        return save_profile(
            {
                "profile_summary": f"Resume uploaded, but automatic parsing failed: {exc}",
                "parsed_profile": {},
                "resume_text": "",
            }
        )
