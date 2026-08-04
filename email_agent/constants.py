"""Shared constants for the email agent."""

from __future__ import annotations

ANALYSIS_STATUSES = frozenset({"Not Analyzed", "Analyzing", "Analyzed", "Relevant", "Not Relevant"})
COMPLETED_ANALYSIS_STATUSES = frozenset({"Analyzed", "Relevant", "Not Relevant"})

RELEVANT_SCORE = 65
HIGH_MATCH_SCORE = 80
SYNC_WINDOW_DAYS = 5

SYNC_PAGE_SIZE = 100
SYNC_MAX_EMAILS = 500

JOB_LABELS = {
    "Not Analyzed": "Job/Not Analyzed",
    "Analyzing": "Job/Analyzing",
    "Analyzed": "Job/Analyzed",
    "Relevant": "Job/Relevant",
    "Not Relevant": "Job/Not Relevant",
}

DEFAULT_JOB_LABEL_STATUS = "Analyzed"

JOB_KEYWORDS = (
    "job",
    "role",
    "position",
    "interview",
    "recruiter",
    "hiring",
    "application",
    "opportunity",
    "resume",
    "cv",
    "candidate",
    "shortlisted",
    "offer",
)
