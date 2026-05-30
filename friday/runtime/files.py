from __future__ import annotations

import json
import mimetypes
import platform
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

TEXT_EXTENSIONS = {
    ".txt",
    ".csv",
    ".json",
    ".md",
    ".log",
    ".xml",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
}


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^\w.\- ]", "_", base).strip(" .")
    return cleaned or "upload"


def normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def is_path_under(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def workdir_relative(workdir: Path, path: Path) -> str:
    return normalize_rel_path(str(path.resolve().relative_to(workdir.resolve())))


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return "text/plain"
    return "application/octet-stream"


def is_text_extension(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def normalize_text_content(content: str) -> str:
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n") and text:
        text += "\n"
    return text


def write_text_file(path: Path, content: str, *, utf8_bom: bool = False) -> int:
    text = normalize_text_content(content)
    encoded = text.encode("utf-8")
    if utf8_bom:
        encoded = b"\xef\xbb\xbf" + encoded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return len(encoded)


def is_preview_image(mime: str, name: str) -> bool:
    if mime.startswith("image/"):
        return True
    ext = Path(name).suffix.lower()
    return ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico"}


def is_preview_text(mime: str, name: str) -> bool:
    if mime.startswith("text/"):
        return True
    ext = Path(name).suffix.lower()
    return ext in TEXT_EXTENSIONS


class FileRegistry:
    def __init__(self, index_path: Optional[Path] = None, workdir: Optional[Path] = None) -> None:
        self._files: Dict[str, Dict[str, Any]] = {}
        self._path_index: Dict[str, str] = {}
        self._index_path = index_path
        self._workdir = workdir.resolve() if workdir else None
        if index_path and index_path.is_file():
            self._load()

    def _load(self) -> None:
        if not self._index_path:
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._files = data.get("files") or {}
        self._path_index = data.get("path_index") or {}
        if not self._workdir:
            return
        stale_ids: List[str] = []
        for file_id, meta in self._files.items():
            abs_path = Path(str(meta.get("abs_path") or ""))
            if not abs_path.is_file() or not is_path_under(self._workdir, abs_path):
                stale_ids.append(file_id)
        for file_id in stale_ids:
            rel = str((self._files.get(file_id) or {}).get("path") or "")
            self._files.pop(file_id, None)
            if self._path_index.get(rel) == file_id:
                self._path_index.pop(rel, None)

    def _save(self) -> None:
        if not self._index_path:
            return
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"files": self._files, "path_index": self._path_index}
        self._index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _meta_from_path(self, path: Path, workdir: Path, *, name: Optional[str] = None) -> Dict[str, Any]:
        resolved = path.resolve()
        display_name = name or resolved.name
        rel_path = workdir_relative(workdir, resolved)
        stat = resolved.stat()
        mime = guess_mime(resolved)
        return {
            "name": display_name,
            "path": rel_path,
            "size": stat.st_size,
            "mime": mime,
            "abs_path": str(resolved),
        }

    def register(
        self,
        path: Path,
        workdir: Path,
        *,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        if not is_path_under(workdir, resolved):
            raise ValueError("path outside workdir")

        base_meta = self._meta_from_path(resolved, workdir, name=name)
        rel_path = base_meta["path"]

        existing_id = self._path_index.get(rel_path)
        if existing_id and existing_id in self._files:
            file_id = existing_id
        else:
            file_id = uuid.uuid4().hex

        meta = {
            "id": file_id,
            "url": f"/api/files/{file_id}",
            **base_meta,
        }
        self._files[file_id] = meta
        self._path_index[rel_path] = file_id
        self._save()
        return {k: v for k, v in meta.items() if k != "abs_path"}

    def get(self, file_id: str) -> Optional[Dict[str, Any]]:
        meta = self._files.get(file_id)
        if not meta:
            return None
        return {k: v for k, v in meta.items() if k != "abs_path"}

    def resolve_download(self, file_id: str, workdir: Path) -> Optional[Path]:
        meta = self._files.get(file_id)
        if not meta:
            return None
        path = Path(meta["abs_path"]).resolve()
        if not path.is_file() or not is_path_under(workdir, path):
            return None
        return path

    def register_existing_paths(
        self,
        workdir: Path,
        rel_dirs: List[str],
    ) -> None:
        for rel_dir in rel_dirs:
            root = (workdir / rel_dir).resolve()
            if not root.is_dir() or not is_path_under(workdir, root):
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    try:
                        self.register(path, workdir)
                    except (OSError, ValueError):
                        continue

    @staticmethod
    def default_utf8_bom() -> bool:
        return platform.system().lower() == "windows"

    def unregister_rel(self, rel_path: str) -> None:
        rel = normalize_rel_path(rel_path)
        file_id = self._path_index.pop(rel, None)
        if file_id:
            self._files.pop(file_id, None)
        self._save()

    def unregister_under(self, rel_prefix: str) -> None:
        prefix = normalize_rel_path(rel_prefix).rstrip("/")
        stale = [
            rel
            for rel in self._path_index
            if rel == prefix or rel.startswith(f"{prefix}/")
        ]
        for rel in stale:
            self.unregister_rel(rel)

    def relocate_file(self, old_rel: str, new_path: Path, workdir: Path) -> None:
        old_key = normalize_rel_path(old_rel)
        file_id = self._path_index.pop(old_key, None)
        if not file_id or file_id not in self._files:
            return
        if not new_path.is_file():
            self._files.pop(file_id, None)
            self._save()
            return
        meta = self._files[file_id]
        updated = self._meta_from_path(new_path, workdir, name=str(meta.get("name") or new_path.name))
        meta.update(updated)
        self._path_index[updated["path"]] = file_id
        self._save()

    def relocate_prefix(self, old_prefix: str, new_prefix: str, workdir: Path) -> None:
        old_p = normalize_rel_path(old_prefix).rstrip("/")
        new_p = normalize_rel_path(new_prefix).rstrip("/")
        updates: List[tuple[str, str, str]] = []
        for rel, file_id in list(self._path_index.items()):
            if rel == old_p or rel.startswith(f"{old_p}/"):
                suffix = rel[len(old_p) :].lstrip("/")
                new_rel = normalize_rel_path(f"{new_p}/{suffix}" if suffix else new_p)
                updates.append((rel, new_rel, file_id))
        for old_rel, new_rel, file_id in updates:
            self._path_index.pop(old_rel, None)
            if file_id not in self._files:
                continue
            new_abs = (workdir / new_rel).resolve()
            if new_abs.is_file() and is_path_under(workdir, new_abs):
                meta = self._files[file_id]
                refreshed = self._meta_from_path(new_abs, workdir, name=str(meta.get("name") or new_abs.name))
                meta.update(refreshed)
                self._path_index[new_rel] = file_id
            else:
                self._files.pop(file_id, None)
        self._save()


def resolve_move_destination(src: Path, dest: Path) -> Path:
    if dest.is_dir():
        return dest / src.name
    if src.is_file() and not dest.suffix and not dest.exists():
        return dest / src.name
    return dest


def move_path_on_disk(
    src: Path,
    dest: Path,
    workdir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    src_resolved = src.resolve()
    dest_resolved = resolve_move_destination(src_resolved, dest.resolve())
    if not src_resolved.exists():
        raise FileNotFoundError(str(src_resolved))
    if not is_path_under(workdir, src_resolved):
        raise ValueError("source outside workdir")
    if not is_path_under(workdir, dest_resolved):
        raise ValueError("destination outside workdir")
    if dest_resolved.exists() and not overwrite:
        raise FileExistsError(str(dest_resolved))
    dest_resolved.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_resolved), str(dest_resolved))
    return dest_resolved.resolve()


def sync_registry_after_move(
    registry: Optional[FileRegistry],
    src_before: Path,
    dest_after: Path,
    workdir: Path,
    *,
    was_file: bool,
) -> None:
    if not registry:
        return
    old_rel = workdir_relative(workdir, src_before)
    if was_file:
        registry.relocate_file(old_rel, dest_after, workdir)
    else:
        registry.relocate_prefix(old_rel, workdir_relative(workdir, dest_after), workdir)
