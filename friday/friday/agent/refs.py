from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import httpx

_REF_PATTERN = re.compile(
    r'@(?:"([^"]+)"|(\S+?))(?=\s|$|[,.;:!?])',
)


def _is_url(token: str) -> bool:
    return token.startswith("http://") or token.startswith("https://")


def _expand_file(workdir: Path, rel: str, max_chars: int) -> str:
    path = (workdir / rel).resolve()
    if path.is_dir():
        return _expand_folder(path, max_chars)
    if not path.is_file():
        return f"(file not found: {rel})"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(read error: {exc})"
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n[... truncated ...]"
    return text


def _expand_folder(path: Path, max_chars: int) -> str:
    if not path.is_dir():
        return "(not a directory)"
    lines = [f"Directory listing: {path.name}/"]
    total = 0
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())[:200]
    except OSError as exc:
        return f"(list error: {exc})"
    for entry in entries:
        kind = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else 0
        line = f"- {entry.name} ({kind}, {size} bytes)"
        if total + len(line) > max_chars:
            lines.append("... (truncated)")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _expand_git_diff(workdir: Path, max_chars: int) -> str:
    parts: List[str] = []
    for args in (["git", "diff"], ["git", "diff", "--staged"]):
        try:
            proc = subprocess.run(
                args,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"(git diff failed: {exc})"
        label = " ".join(args[1:])
        out = (proc.stdout or "").strip()
        if out:
            parts.append(f"## {label}\n{out}")
    if not parts:
        return "(no git diff output)"
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n[... truncated ...]"
    return text


def _expand_url(url: str, max_chars: int, allow_fetch: bool) -> str:
    if not allow_fetch:
        return "(URL fetch disabled; set REFS_ALLOW_URL_FETCH=true)"
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            text = resp.text
    except Exception as exc:
        return f"(fetch error: {exc})"
    if len(text) > max_chars:
        text = text[: max_chars - 20] + "\n[... truncated ...]"
    return text


def expand_references(
    message: str,
    workdir: Path,
    *,
    max_file_chars: int,
    max_total_chars: int,
    allow_url_fetch: bool,
) -> Tuple[str, bool]:
    tokens: List[Tuple[str, str]] = []
    for match in _REF_PATTERN.finditer(message):
        quoted, plain = match.group(1), match.group(2)
        token = (quoted or plain or "").strip()
        if not token:
            continue
        tokens.append((match.group(0), token))
    if not tokens:
        return message, False
    blocks: List[str] = ["## Referenced context"]
    used = 0
    for ref_token, path_or_url in tokens:
        if path_or_url.lower() in {"diff", "git-diff", "git_diff"}:
            body = _expand_git_diff(workdir, max_file_chars)
        elif _is_url(path_or_url):
            body = _expand_url(path_or_url, max_file_chars, allow_url_fetch)
        elif path_or_url.endswith("/"):
            body = _expand_folder((workdir / path_or_url.rstrip("/")).resolve(), max_file_chars)
        else:
            body = _expand_file(workdir, path_or_url, max_file_chars)
        block = f"### {ref_token}\n{body}"
        if used + len(block) > max_total_chars:
            block = block[: max_total_chars - used - 20] + "\n[... truncated ...]"
        blocks.append(block)
        used += len(block)
        if used >= max_total_chars:
            break
    expanded = message + "\n\n" + "\n\n".join(blocks)
    return expanded, True
