from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from config import settings
from llm import as_list, chat_completion_json

logger = logging.getLogger(__name__)


SKILL_CATALOG = (
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "go",
    "rust",
    "sql",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "linux",
    "fastapi",
    "django",
    "flask",
    "react",
    "node.js",
    "express",
    "machine learning",
    "generative ai",
    "genai",
    "deep learning",
    "nlp",
    "llm",
    "rag",
    "agentic ai",
    "mcp",
    "semantic kernel",
    "langchain",
    "langgraph",
    "transformers",
    "hugging face",
    "computer vision",
    "yolo",
    "opencv",
    "ocr",
    "whisper",
    "robotics",
    "ros",
    "ros2",
    "slam",
    "sensor fusion",
    "jetson",
    "arduino",
    "esp32",
    "mlops",
    "websockets",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "data analysis",
    "data engineering",
    "api",
    "rest",
    "graphql",
    "git",
    "ci/cd",
)


def parse_resume_file(path: Path, current_profile: dict[str, Any]) -> dict[str, Any]:
    text = extract_resume_text(path)
    parsed = fallback_parse_resume(text)
    if settings.llm_configured and text:
        try:
            llm_parsed = llm_parse_resume(text, current_profile)
            if _has_useful_profile(llm_parsed):
                parsed = _merge_parsed_profiles(parsed, llm_parsed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM resume parse failed; using fallback: %s", exc)
            parsed = fallback_parse_resume(text)
    return normalize_parsed_resume(parsed, text, current_profile)


def extract_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix in {".txt", ".md", ".rtf"}:
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        return ""


def llm_parse_resume(text: str, current_profile: dict[str, Any]) -> dict[str, Any]:
    prompt = {
        "resume_text": text[:18000],
        "existing_profile": {
            "full_name": current_profile.get("full_name"),
            "email": current_profile.get("email"),
            "phone": current_profile.get("phone"),
            "location": current_profile.get("location"),
            "linkedin": current_profile.get("linkedin"),
            "portfolio": current_profile.get("portfolio"),
            "job_preferences": current_profile.get("job_preferences"),
        },
    }
    return chat_completion_json(
        [
            {
                "role": "system",
                "content": (
                    "Extract a candidate profile from a resume. Return strict JSON only with keys: "
                    "full_name, email, phone, location, linkedin, portfolio, summary, work_experience, "
                    "education, skills, tools, certifications, job_preferences, natural_language_summary. "
                    "Use arrays for skills, tools, certifications, and education. Do not invent facts."
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=0.1,
    )


def fallback_parse_resume(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    email = _first_match(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone = _first_match(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
    links = re.findall(r"https?://[^\s)]+|(?:linkedin\.com|github\.com|[\w.-]+\.[a-z]{2,})(?:/[^\s)]*)?", text, flags=re.I)
    linkedin = next((link for link in links if "linkedin.com" in link.lower()), "")
    portfolio = next((link for link in links if link != linkedin), "")
    skills = _extract_known_skills(text)
    name = _guess_name(lines, email)
    summary = _section_text(text, ("executive summary", "summary", "profile")) or (
        " ".join(lines[1:4]) if len(lines) > 1 else (lines[0] if lines else "")
    )
    return {
        "full_name": name,
        "email": email,
        "phone": phone,
        "location": "",
        "linkedin": linkedin,
        "portfolio": portfolio,
        "summary": summary[:700],
        "work_experience": _section_text(text, ("work experience", "professional experience", "experience", "work history", "employment")),
        "education": _section_list(text, ("education & achievements", "education")),
        "skills": skills,
        "tools": [],
        "certifications": _section_list(text, ("certifications", "certificates")),
        "job_preferences": "",
        "natural_language_summary": _natural_summary(name, summary, skills),
    }


def normalize_parsed_resume(parsed: dict[str, Any], text: str, current_profile: dict[str, Any]) -> dict[str, Any]:
    structured = {
        "full_name": _as_text(parsed.get("full_name")),
        "email": _as_text(parsed.get("email")),
        "phone": _as_text(parsed.get("phone")),
        "location": _as_text(parsed.get("location")),
        "linkedin": _as_text(parsed.get("linkedin")),
        "portfolio": _as_text(parsed.get("portfolio")),
        "summary": _as_text(parsed.get("summary")),
        "work_experience": _as_text(parsed.get("work_experience")),
        "education": _as_list(parsed.get("education")),
        "skills": _as_list(parsed.get("skills")),
        "tools": _as_list(parsed.get("tools")),
        "certifications": _as_list(parsed.get("certifications")),
        "job_preferences": _as_text(parsed.get("job_preferences")),
        "natural_language_summary": _as_text(parsed.get("natural_language_summary")),
    }
    if not structured["natural_language_summary"]:
        structured["natural_language_summary"] = _natural_summary(
            structured["full_name"] or current_profile.get("full_name", ""),
            structured["summary"],
            structured["skills"],
        )
    return {
        "resume_text": text,
        "parsed_profile": structured,
        "profile_summary": structured["natural_language_summary"],
    }


def merge_resume_profile(profile: dict[str, Any], parsed_data: dict[str, Any]) -> dict[str, Any]:
    structured = parsed_data.get("parsed_profile") or {}
    merged = profile.copy()
    for key in ("full_name", "email", "phone", "location", "linkedin", "portfolio", "summary", "work_experience", "job_preferences"):
        if not merged.get(key) and structured.get(key):
            value = structured[key]
            merged[key] = "\n".join(value) if isinstance(value, list) else value
    parsed_skills = structured.get("skills") or []
    if parsed_skills and not merged.get("skills"):
        merged["skills"] = ", ".join(parsed_skills)
    merged["resume_text"] = parsed_data.get("resume_text", "")
    merged["parsed_profile"] = structured
    merged["profile_summary"] = parsed_data.get("profile_summary", "")
    return merged


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return ""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _extract_docx_text(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        return ""
    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()


def _first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""


def _extract_known_skills(text: str) -> list[str]:
    lower = text.lower()
    found = [skill for skill in SKILL_CATALOG if re.search(rf"(?<![\w+#.-]){re.escape(skill)}(?![\w+#.-])", lower)]
    return sorted(set(found), key=found.index)


def _guess_name(lines: list[str], email: str) -> str:
    leading_words: list[str] = []
    for line in lines[:12]:
        if re.fullmatch(r"[A-Z][A-Z.'-]+", line):
            leading_words.append(line.title())
            if len(leading_words) == 4:
                break
            continue
        if leading_words:
            break
    if len(leading_words) >= 2:
        return " ".join(leading_words)
    for line in lines[:6]:
        if email and email in line:
            continue
        if len(line.split()) in {2, 3, 4} and not re.search(r"\d|@|https?://", line):
            return line
    return ""


def _section_text(text: str, headings: tuple[str, ...]) -> str:
    sections = _sections(text)
    for heading in headings:
        for key, value in sections.items():
            if heading in key:
                return value[:3000]
    return ""


def _section_list(text: str, headings: tuple[str, ...]) -> list[str]:
    section = _section_text(text, headings)
    return [line.strip(" -•\t") for line in section.splitlines() if line.strip()][:8]


def _sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current = "intro"
    sections[current] = []
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) < 40 and stripped.lower().strip(":") in {
            "summary",
            "executive summary",
            "profile",
            "experience",
            "work experience",
            "professional experience",
            "work history",
            "employment",
            "education",
            "education & achievements",
            "skills",
            "technical skills",
            "certifications",
            "certificates",
            "patents & innovation",
            "projects",
        }:
            current = stripped.lower().strip(":")
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item).strip() for item in value if str(item).strip())
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _as_list(value: Any) -> list[str]:
    return as_list(value)


def _has_useful_profile(parsed: dict[str, Any]) -> bool:
    if not isinstance(parsed, dict):
        return False
    scalar_fields = ("full_name", "email", "phone", "summary", "work_experience")
    if any(str(parsed.get(field) or "").strip() for field in scalar_fields):
        return True
    return any(_as_list(parsed.get(field)) for field in ("skills", "tools", "education", "certifications"))


def _merge_parsed_profiles(fallback: dict[str, Any], preferred: dict[str, Any]) -> dict[str, Any]:
    merged = fallback.copy()
    list_fields = {"education", "skills", "tools", "certifications"}
    for key, value in preferred.items():
        if key in list_fields:
            merged[key] = _merge_lists(_as_list(fallback.get(key)), _as_list(value))
        elif str(value or "").strip():
            merged[key] = value
    return merged


def _merge_lists(left: list[str], right: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in [*right, *left]:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _natural_summary(name: str, summary: str, skills: list[str]) -> str:
    display_name = name or "This candidate"
    skill_text = ", ".join(skills[:8])
    if summary and skill_text:
        return f"{display_name} has experience described as: {summary[:320]} Key skills include {skill_text}."
    if skill_text:
        return f"{display_name} has a profile centered on {skill_text}."
    if summary:
        return f"{display_name} has the following professional summary: {summary[:420]}"
    return "Upload a resume or add profile details to build a richer candidate profile."
