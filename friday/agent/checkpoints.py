from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from friday.runtime.files import is_path_under
from friday.runtime.persist import atomic_write_json


@dataclass
class CheckpointMeta:
    id: str
    created_at: float
    tool: str
    paths: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    def __init__(self, root: Path, workdir: Path, max_count: int, max_file_bytes: int) -> None:
        self.root = root.resolve()
        self.workdir = workdir.resolve()
        self.max_count = max(1, max_count)
        self.max_file_bytes = max_file_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self._index_path.is_file():
            return []
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            return list(data) if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_index(self, items: List[Dict[str, Any]]) -> None:
        atomic_write_json(self._index_path, items[-self.max_count :])

    def snapshot_before(self, tool: str, paths: List[Path]) -> Optional[str]:
        copied: List[str] = []
        cp_id = uuid.uuid4().hex[:12]
        cp_dir = self.root / cp_id
        cp_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            src = src.resolve()
            if not src.is_file() or not is_path_under(self.workdir, src):
                continue
            if src.stat().st_size > self.max_file_bytes:
                continue
            rel = str(src.relative_to(self.workdir))
            dest = cp_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel)
        if not copied:
            shutil.rmtree(cp_dir, ignore_errors=True)
            return None
        meta = CheckpointMeta(id=cp_id, created_at=time.time(), tool=tool, paths=copied)
        atomic_write_json(cp_dir / "meta.json", meta.to_dict())
        index = self._load_index()
        index.append(meta.to_dict())
        self._save_index(index)
        dirs = sorted(
            (p for p in self.root.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        while len(dirs) > self.max_count:
            shutil.rmtree(dirs.pop(0), ignore_errors=True)
        return cp_id

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return list(reversed(self._load_index()))

    def rollback(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        index = self._load_index()
        if not index:
            return {"ok": False, "error": "no checkpoints"}
        if checkpoint_id in (None, "", "last"):
            meta = index[-1]
        else:
            meta = next((m for m in index if m.get("id") == checkpoint_id), None)
            if not meta:
                return {"ok": False, "error": f"checkpoint not found: {checkpoint_id}"}
        cp_id = str(meta.get("id"))
        cp_dir = self.root / cp_id
        if not cp_dir.is_dir():
            return {"ok": False, "error": "checkpoint data missing"}
        restored: List[str] = []
        for rel in meta.get("paths") or []:
            snap = cp_dir / rel
            target = self.workdir / rel
            if snap.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snap, target)
                restored.append(rel)
        return {"ok": True, "checkpoint_id": cp_id, "restored": restored}
