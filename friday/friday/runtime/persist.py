from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve()).lower()
    with _locks_guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _commit_tmp(tmp: Path, dest: Path) -> None:
    """Move tmp into place; Windows often denies replace() on existing files."""
    if sys.platform == "win32" and dest.exists():
        dest.unlink()
    os.replace(str(tmp), str(dest))


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically with retries and a Windows-safe fallback."""
    dest = path.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock = _path_lock(dest)
    with lock:
        last_error: Optional[OSError] = None
        for attempt in range(8):
            tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{attempt}.tmp")
            try:
                tmp.write_text(content, encoding=encoding)
                _commit_tmp(tmp, dest)
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
            finally:
                tmp.unlink(missing_ok=True)

        try:
            dest.write_text(content, encoding=encoding)
            logger.warning("atomic write fell back to direct write: %s", dest)
            return
        except OSError as exc:
            logger.error("failed to persist %s: %s", dest, exc)
            if last_error is not None:
                raise last_error from exc
            raise


def atomic_write_json(path: Path, data: Any, **json_kwargs: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, **json_kwargs)
    atomic_write_text(path, text)
