from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from config import DB_FILE, ensure_directories
from constants import (
    ANALYSIS_STATUSES,
    COMPLETED_ANALYSIS_STATUSES,
    HIGH_MATCH_SCORE,
    SYNC_WINDOW_DAYS,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterable[sqlite3.Connection]:
    ensure_directories()
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS emails (
                gmail_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                history_id TEXT,
                sender TEXT,
                sender_email TEXT,
                recipient TEXT,
                subject TEXT,
                snippet TEXT,
                body_text TEXT,
                body_html TEXT,
                received_at TEXT,
                internal_date INTEGER,
                is_job INTEGER DEFAULT 0,
                is_job_email INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0,
                status TEXT DEFAULT 'Not Analyzed',
                analysis_status TEXT DEFAULT 'Not Analyzed',
                company TEXT,
                role TEXT,
                job_title TEXT,
                job_type TEXT,
                match_score INTEGER DEFAULT 0,
                required_skills_json TEXT DEFAULT '[]',
                matched_skills_json TEXT DEFAULT '[]',
                missing_skills_json TEXT DEFAULT '[]',
                action_needed TEXT,
                analysis_json TEXT,
                last_analyzed_at TEXT,
                analysis_version INTEGER DEFAULT 0,
                labels_json TEXT DEFAULT '[]',
                archived INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gmail_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                attach_resume INTEGER DEFAULT 0,
                status TEXT DEFAULT 'draft',
                sent_message_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (gmail_id) REFERENCES emails(gmail_id)
            );

            CREATE TABLE IF NOT EXISTS gmail_labels (
                name TEXT PRIMARY KEY,
                gmail_label_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sync_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_synced_at TEXT,
                last_internal_date INTEGER DEFAULT 0,
                last_gmail_id TEXT
            );

            CREATE TABLE IF NOT EXISTS resume_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_path TEXT NOT NULL,
                file_name TEXT,
                full_name TEXT,
                email TEXT,
                phone TEXT,
                location TEXT,
                linkedin TEXT,
                portfolio TEXT,
                summary TEXT,
                work_experience TEXT,
                job_preferences TEXT,
                natural_language_summary TEXT,
                resume_text TEXT,
                education_json TEXT DEFAULT '[]',
                skills_json TEXT DEFAULT '[]',
                tools_json TEXT DEFAULT '[]',
                certifications_json TEXT DEFAULT '[]',
                parsed_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _ensure_email_columns(db)
        db.execute(
            """
            UPDATE emails
            SET status='Not Analyzed', analysis_status='Not Analyzed'
            WHERE status NOT IN ('Not Analyzed', 'Analyzing', 'Analyzed', 'Relevant', 'Not Relevant')
            """
        )


def _ensure_email_columns(db: sqlite3.Connection) -> None:
    existing = {row["name"] for row in db.execute("PRAGMA table_info(emails)").fetchall()}
    columns = {
        "is_job_email": "INTEGER DEFAULT 0",
        "analysis_status": "TEXT DEFAULT 'Not Analyzed'",
        "job_title": "TEXT",
        "job_type": "TEXT",
        "match_score": "INTEGER DEFAULT 0",
        "required_skills_json": "TEXT DEFAULT '[]'",
        "matched_skills_json": "TEXT DEFAULT '[]'",
        "missing_skills_json": "TEXT DEFAULT '[]'",
        "last_analyzed_at": "TEXT",
        "analysis_version": "INTEGER DEFAULT 0",
        "body_html": "TEXT",
    }
    for name, declaration in columns.items():
        if name not in existing:
            db.execute(f"ALTER TABLE emails ADD COLUMN {name} {declaration}")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for field in ("analysis_json", "labels_json", "required_skills_json", "matched_skills_json", "missing_skills_json"):
        raw = data.get(field)
        if raw:
            try:
                data[field] = json.loads(raw)
            except json.JSONDecodeError:
                data[field] = {} if field == "analysis_json" else []
    data["status"] = data.get("analysis_status") or data.get("status") or "Not Analyzed"
    data["analysis_status"] = data["status"]
    data["is_job_email"] = bool(data.get("is_job_email") or data.get("is_job"))
    data["is_job"] = data["is_job_email"]
    data["match_score"] = int(data.get("match_score") or round(float(data.get("confidence") or 0) * 100))
    data["job_title"] = data.get("job_title") or data.get("role") or ""
    data["required_skills"] = data.get("required_skills_json") or []
    data["matched_skills"] = data.get("matched_skills_json") or []
    data["missing_skills"] = data.get("missing_skills_json") or []
    return data


def email_exists(gmail_id: str) -> bool:
    with connect() as db:
        row = db.execute("SELECT 1 FROM emails WHERE gmail_id = ?", (gmail_id,)).fetchone()
    return row is not None


def latest_synced_email() -> dict[str, Any] | None:
    with connect() as db:
        return row_to_dict(
            db.execute(
                "SELECT * FROM emails ORDER BY COALESCE(internal_date, 0) DESC LIMIT 1"
            ).fetchone()
        )


def last_analyzed_email() -> dict[str, Any] | None:
    with connect() as db:
        return row_to_dict(
            db.execute(
                """
                SELECT * FROM emails
                WHERE last_analyzed_at IS NOT NULL AND last_analyzed_at != ''
                ORDER BY COALESCE(internal_date, 0) DESC
                LIMIT 1
                """
            ).fetchone()
        )


def save_sync_state(gmail_id: str | None, internal_date: int | None) -> None:
    now = utc_now()
    with connect() as db:
        db.execute(
            """
            INSERT INTO sync_state (id, last_synced_at, last_internal_date, last_gmail_id)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_synced_at=excluded.last_synced_at,
                last_internal_date=excluded.last_internal_date,
                last_gmail_id=excluded.last_gmail_id
            """,
            (now, int(internal_date or 0), gmail_id or ""),
        )


def get_sync_state() -> dict[str, Any]:
    with connect() as db:
        row = db.execute("SELECT * FROM sync_state WHERE id = 1").fetchone()
    if not row:
        return {"last_synced_at": None, "last_internal_date": 0, "last_gmail_id": ""}
    return dict(row)


def prune_emails_older_than_days(days: int = SYNC_WINDOW_DAYS) -> int:
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    with connect() as db:
        cursor = db.execute(
            "DELETE FROM emails WHERE COALESCE(internal_date, 0) > 0 AND internal_date < ?",
            (cutoff_ms,),
        )
    return int(cursor.rowcount)


def get_sync_info() -> dict[str, Any]:
    latest = latest_synced_email()
    last_analyzed = last_analyzed_email()
    sync_state = get_sync_state()
    metrics = dashboard_metrics()
    return {
        "window_days": SYNC_WINDOW_DAYS,
        "total_in_db": metrics["total_scanned"],
        "pending_analysis": metrics["pending_analysis"],
        "last_synced_at": sync_state.get("last_synced_at"),
        "latest_email_at": latest.get("received_at") if latest else None,
        "latest_email_subject": latest.get("subject") if latest else None,
        "last_analyzed_at": last_analyzed.get("last_analyzed_at") if last_analyzed else None,
        "last_analyzed_subject": last_analyzed.get("subject") if last_analyzed else None,
    }


def upsert_email(message: dict[str, Any]) -> None:
    now = utc_now()
    labels = json.dumps(message.get("label_ids", []))
    with connect() as db:
        db.execute(
            """
            INSERT INTO emails (
                gmail_id, thread_id, history_id, sender, sender_email, recipient,
                subject, snippet, body_text, body_html, received_at, internal_date, labels_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gmail_id) DO UPDATE SET
                thread_id=excluded.thread_id,
                history_id=excluded.history_id,
                sender=excluded.sender,
                sender_email=excluded.sender_email,
                recipient=excluded.recipient,
                subject=excluded.subject,
                snippet=excluded.snippet,
                body_text=excluded.body_text,
                body_html=excluded.body_html,
                received_at=excluded.received_at,
                internal_date=excluded.internal_date,
                labels_json=excluded.labels_json,
                updated_at=excluded.updated_at
            """,
            (
                message["id"],
                message.get("thread_id", ""),
                message.get("history_id"),
                message.get("sender"),
                message.get("sender_email"),
                message.get("recipient"),
                message.get("subject", ""),
                message.get("snippet", ""),
                message.get("body_text", ""),
                message.get("body_html", ""),
                message.get("received_at"),
                message.get("internal_date"),
                labels,
                now,
                now,
            ),
        )


def save_analysis(gmail_id: str, analysis: dict[str, Any]) -> None:
    status = analysis.get("status") if analysis.get("status") in ANALYSIS_STATUSES else "Analyzed"
    match_score = int(analysis.get("match_score") or round(float(analysis.get("confidence") or 0) * 100))
    is_job_email = bool(analysis.get("is_job_email", analysis.get("is_job")))
    with connect() as db:
        db.execute(
            """
            UPDATE emails
            SET is_job=?, is_job_email=?, confidence=?, status=?, analysis_status=?,
                company=?, role=?, job_title=?, job_type=?, match_score=?,
                required_skills_json=?, matched_skills_json=?, missing_skills_json=?,
                action_needed=?, analysis_json=?, last_analyzed_at=?, analysis_version=?,
                updated_at=?
            WHERE gmail_id=?
            """,
            (
                1 if is_job_email else 0,
                1 if is_job_email else 0,
                match_score / 100,
                status,
                status,
                analysis.get("company") or "",
                analysis.get("role") or analysis.get("job_title") or "",
                analysis.get("job_title") or analysis.get("role") or "",
                analysis.get("job_type") or "",
                match_score,
                json.dumps(analysis.get("required_skills") or []),
                json.dumps(analysis.get("matched_skills") or []),
                json.dumps(analysis.get("missing_skills") or []),
                analysis.get("recommended_action") or analysis.get("action_needed") or "",
                json.dumps(analysis),
                utc_now(),
                int(analysis.get("analysis_version") or 1),
                utc_now(),
                gmail_id,
            ),
        )


def set_analysis_status(gmail_id: str, status: str) -> None:
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"Invalid analysis status: {status}")
    with connect() as db:
        db.execute(
            "UPDATE emails SET status=?, analysis_status=?, updated_at=? WHERE gmail_id=?",
            (status, status, utc_now(), gmail_id),
        )


def should_analyze(gmail_id: str, force: bool = False) -> bool:
    if force:
        return True
    email = get_email(gmail_id)
    if not email:
        return False
    return email.get("analysis_status") not in COMPLETED_ANALYSIS_STATUSES


def list_emails(
    status: str | None = None,
    include_non_jobs: bool = True,
    limit: int = 100,
    min_score: int | None = None,
    company: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    job_type: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT e.*,
            (
                SELECT d.status
                FROM drafts d
                WHERE d.gmail_id = e.gmail_id
                ORDER BY d.id DESC
                LIMIT 1
            ) AS latest_draft_status
        FROM emails e
    """
    values: list[Any] = []
    clauses: list[str] = []
    if not include_non_jobs:
        clauses.append("e.is_job_email = 1")
    if status and status != "All":
        clauses.append("e.analysis_status = ?")
        values.append(status)
    if min_score is not None:
        clauses.append("e.match_score >= ?")
        values.append(min_score)
    if company:
        clauses.append("LOWER(e.company) LIKE ?")
        values.append(f"%{company.lower()}%")
    if job_type:
        clauses.append("LOWER(e.job_type) LIKE ?")
        values.append(f"%{job_type.lower()}%")
    if date_from:
        clauses.append("date(e.received_at) >= date(?)")
        values.append(date_from)
    if date_to:
        clauses.append("date(e.received_at) <= date(?)")
        values.append(date_to)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY COALESCE(e.internal_date, 0) DESC LIMIT ?"
    values.append(limit)
    with connect() as db:
        emails = [row_to_dict(row) for row in db.execute(query, values).fetchall() if row]
    for email in emails:
        # Keep list payloads light; full body is loaded in get_email.
        email.pop("body_html", None)
        email.pop("body_text", None)
    return emails


def get_email(gmail_id: str) -> dict[str, Any] | None:
    with connect() as db:
        return row_to_dict(db.execute("SELECT * FROM emails WHERE gmail_id = ?", (gmail_id,)).fetchone())


def save_draft(gmail_id: str, subject: str, body: str, attach_resume: bool) -> dict[str, Any]:
    now = utc_now()
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO drafts (gmail_id, subject, body, attach_resume, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'draft', ?, ?)
            """,
            (gmail_id, subject, body, 1 if attach_resume else 0, now, now),
        )
        draft_id = cursor.lastrowid
    return get_draft(int(draft_id)) or {}


def update_draft(draft_id: int, subject: str, body: str, attach_resume: bool) -> dict[str, Any] | None:
    with connect() as db:
        db.execute(
            """
            UPDATE drafts
            SET subject=?, body=?, attach_resume=?, updated_at=?
            WHERE id=? AND status = 'draft'
            """,
            (subject, body, 1 if attach_resume else 0, utc_now(), draft_id),
        )
    return get_draft(draft_id)


def get_draft(draft_id: int) -> dict[str, Any] | None:
    with connect() as db:
        return row_to_dict(db.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,)).fetchone())


def latest_draft(gmail_id: str) -> dict[str, Any] | None:
    with connect() as db:
        return row_to_dict(
            db.execute(
                "SELECT * FROM drafts WHERE gmail_id = ? ORDER BY id DESC LIMIT 1",
                (gmail_id,),
            ).fetchone()
        )


def mark_draft_sent(draft_id: int, sent_message_id: str) -> None:
    with connect() as db:
        db.execute(
            "UPDATE drafts SET status='sent', sent_message_id=?, updated_at=? WHERE id=?",
            (sent_message_id, utc_now(), draft_id),
        )


def update_email_status(gmail_id: str, status: str) -> None:
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"Invalid analysis status: {status}")
    with connect() as db:
        db.execute(
            "UPDATE emails SET status=?, analysis_status=?, updated_at=? WHERE gmail_id=?",
            (status, status, utc_now(), gmail_id),
        )


def mark_archived(gmail_id: str) -> None:
    with connect() as db:
        db.execute(
            "UPDATE emails SET archived=1, updated_at=? WHERE gmail_id=?",
            (utc_now(), gmail_id),
        )


def save_label(name: str, gmail_label_id: str) -> None:
    with connect() as db:
        db.execute(
            """
            INSERT INTO gmail_labels (name, gmail_label_id, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET gmail_label_id=excluded.gmail_label_id, updated_at=excluded.updated_at
            """,
            (name, gmail_label_id, utc_now()),
        )


def mark_all_emails_pending_analysis() -> int:
    with connect() as db:
        cursor = db.execute(
            """
            UPDATE emails
            SET status='Not Analyzed', analysis_status='Not Analyzed', updated_at=?
            """,
            (utc_now(),),
        )
    return int(cursor.rowcount or 0)


def pending_email_ids(limit: int = 100) -> list[str]:
    with connect() as db:
        rows = db.execute(
            """
            SELECT gmail_id FROM emails
            WHERE analysis_status IN ('Not Analyzed', 'Analyzing')
            ORDER BY COALESCE(internal_date, 0) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row["gmail_id"] for row in rows]


def count_pending_emails() -> int:
    with connect() as db:
        row = db.execute(
            """
            SELECT COUNT(*) AS total FROM emails
            WHERE analysis_status IN ('Not Analyzed', 'Analyzing')
            """
        ).fetchone()
    return int(row["total"] or 0)


def dashboard_metrics() -> dict[str, int]:
    with connect() as db:
        row = db.execute(
            """
            SELECT
                COUNT(*) AS total_scanned,
                SUM(CASE WHEN analysis_status = 'Relevant' THEN 1 ELSE 0 END) AS relevant_job_emails,
                SUM(CASE WHEN analysis_status = 'Not Relevant' THEN 1 ELSE 0 END) AS non_job_emails,
                SUM(CASE WHEN analysis_status IN ('Not Analyzed', 'Analyzing') THEN 1 ELSE 0 END) AS pending_analysis,
                SUM(CASE WHEN match_score >= ? THEN 1 ELSE 0 END) AS high_match_opportunities
            FROM emails
            """,
            (HIGH_MATCH_SCORE,),
        ).fetchone()
    return {
        "total_scanned": int(row["total_scanned"] or 0),
        "relevant_job_emails": int(row["relevant_job_emails"] or 0),
        "non_job_emails": int(row["non_job_emails"] or 0),
        "pending_analysis": int(row["pending_analysis"] or 0),
        "high_match_opportunities": int(row["high_match_opportunities"] or 0),
    }


def filter_options() -> dict[str, list[str]]:
    with connect() as db:
        companies = [
            row["company"]
            for row in db.execute(
                "SELECT DISTINCT company FROM emails WHERE company IS NOT NULL AND company != '' ORDER BY company LIMIT 200"
            ).fetchall()
        ]
        job_types = [
            row["job_type"]
            for row in db.execute(
                "SELECT DISTINCT job_type FROM emails WHERE job_type IS NOT NULL AND job_type != '' ORDER BY job_type LIMIT 100"
            ).fetchall()
        ]
    return {"companies": companies, "job_types": job_types}


def save_resume_profile(resume_path: str, parsed_data: dict[str, Any]) -> dict[str, Any]:
    init_db()
    structured = parsed_data.get("parsed_profile") or {}
    now = utc_now()
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO resume_profiles (
                resume_path, file_name, full_name, email, phone, location, linkedin,
                portfolio, summary, work_experience, job_preferences,
                natural_language_summary, resume_text, education_json, skills_json,
                tools_json, certifications_json, parsed_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resume_path,
                str(resume_path).replace("\\", "/").split("/")[-1],
                structured.get("full_name") or "",
                structured.get("email") or "",
                structured.get("phone") or "",
                structured.get("location") or "",
                structured.get("linkedin") or "",
                structured.get("portfolio") or "",
                structured.get("summary") or "",
                structured.get("work_experience") or "",
                structured.get("job_preferences") or "",
                structured.get("natural_language_summary") or parsed_data.get("profile_summary") or "",
                parsed_data.get("resume_text") or "",
                json.dumps(structured.get("education") or []),
                json.dumps(structured.get("skills") or []),
                json.dumps(structured.get("tools") or []),
                json.dumps(structured.get("certifications") or []),
                json.dumps(parsed_data),
                now,
                now,
            ),
        )
        profile_id = int(cursor.lastrowid)
    return get_resume_profile(profile_id) or {}


def get_resume_profile(profile_id: int) -> dict[str, Any] | None:
    with connect() as db:
        return resume_profile_row_to_dict(
            db.execute("SELECT * FROM resume_profiles WHERE id = ?", (profile_id,)).fetchone()
        )


def latest_resume_profile() -> dict[str, Any] | None:
    with connect() as db:
        return resume_profile_row_to_dict(
            db.execute("SELECT * FROM resume_profiles ORDER BY id DESC LIMIT 1").fetchone()
        )


def latest_resume_profile_for_path(resume_path: str) -> dict[str, Any] | None:
    with connect() as db:
        return resume_profile_row_to_dict(
            db.execute(
                "SELECT * FROM resume_profiles WHERE resume_path = ? ORDER BY id DESC LIMIT 1",
                (resume_path,),
            ).fetchone()
        )


def resume_profile_row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for field in ("education_json", "skills_json", "tools_json", "certifications_json", "parsed_json"):
        raw = data.get(field)
        try:
            data[field] = json.loads(raw) if raw else ([] if field != "parsed_json" else {})
        except json.JSONDecodeError:
            data[field] = [] if field != "parsed_json" else {}
    data["education"] = data.get("education_json") or []
    data["skills"] = data.get("skills_json") or []
    data["tools"] = data.get("tools_json") or []
    data["certifications"] = data.get("certifications_json") or []
    data["parsed_profile"] = data.get("parsed_json", {}).get("parsed_profile", {})
    return data
