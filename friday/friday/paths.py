from __future__ import annotations

from pathlib import Path

# Project directory: this file lives at <repo>/friday/friday/paths.py, so the
# project root (containing setup.py, .env, soul.md, .friday/) is two levels up.
BASE_DIR = Path(__file__).resolve().parent.parent
FRIDAY_DIR = BASE_DIR / ".friday"

FRIDAY_DIR.mkdir(parents=True, exist_ok=True)

TASK_FILE = FRIDAY_DIR / "task_queue.json"
AUTONOMY_STATE_FILE = FRIDAY_DIR / "autonomy_state.json"
CONVERSATION_FILE = FRIDAY_DIR / "conversation.json"
HOOKS_FILE = FRIDAY_DIR / "hooks.json"
FILE_REGISTRY_FILE = FRIDAY_DIR / "file_registry.json"
CRON_JOBS_FILE = FRIDAY_DIR / "cron_jobs.json"
TOKEN_USAGE_FILE = FRIDAY_DIR / "token_usage.json"
CHECKPOINTS_DIR = FRIDAY_DIR / "checkpoints"
BATCHES_DIR = FRIDAY_DIR / "batches"
SANDBOX_DIR = FRIDAY_DIR / "sandbox"
SOUL_FILE = BASE_DIR / "soul.md"
