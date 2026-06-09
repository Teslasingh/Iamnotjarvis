"""OCR via pytesseract — supplements DOM/vision for labels, buttons, and form text."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_OCR_AVAILABLE: Optional[bool] = None
_MAX_OCR_CHARS = int(os.getenv("OCR_MAX_CHARS", "3500"))
_MIN_CONFIDENCE = int(os.getenv("OCR_MIN_CONFIDENCE", "35"))
_OCR_CACHE_MAX = int(os.getenv("OCR_CACHE_SIZE", "12"))
_ocr_line_cache: Dict[str, List["OcrLine"]] = {}


def _screenshot_cache_key(b64_data: str) -> str:
    sample = b64_data[:12000] if len(b64_data) > 12000 else b64_data
    return hashlib.sha256(sample.encode("utf-8", errors="ignore")).hexdigest()[:20]


def _cache_ocr_lines(key: str, lines: List["OcrLine"]) -> List["OcrLine"]:
    if key and lines:
        _ocr_line_cache[key] = lines
        while len(_ocr_line_cache) > _OCR_CACHE_MAX:
            _ocr_line_cache.pop(next(iter(_ocr_line_cache)))
    return lines

_FORM_KEYWORDS = re.compile(
    r"\*|required|mandatory|submit|apply|next|continue|search|email|phone|"
    r"resume|upload|salary|notice|remote|experience|password|sign\s*in",
    re.IGNORECASE,
)


@dataclass
class OcrLine:
    text: str
    confidence: float
    top: int
    left: int
    height: int
    width: int

    @property
    def vertical_center(self) -> int:
        return self.top + self.height // 2


def find_tesseract_executable() -> Optional[str]:
    """
    Locate tesseract.exe / tesseract binary.

    pytesseract is only a Python wrapper; the UB Mannheim / system Tesseract
    program must exist separately (often missing from PATH on Windows).
    """
    env_cmd = os.getenv("TESSERACT_CMD", "").strip().strip('"')
    if env_cmd:
        path = Path(env_cmd)
        if path.is_file():
            return str(path.resolve())
        logger.warning("TESSERACT_CMD is set but not found: %s", env_cmd)

    found = shutil.which("tesseract")
    if found:
        return found

    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        for candidate in (
            Path(pf) / "Tesseract-OCR" / "tesseract.exe",
            Path(pf86) / "Tesseract-OCR" / "tesseract.exe",
            Path(local) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
            Path(local) / "Tesseract-OCR" / "tesseract.exe",
        ):
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def _configure_tesseract() -> Optional[str]:
    """Point pytesseract at the binary; return resolved path or None."""
    import pytesseract

    cmd = find_tesseract_executable()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd


def reset_ocr_availability_cache() -> None:
    """Re-run detection (e.g. after fixing TESSERACT_CMD)."""
    global _OCR_AVAILABLE
    _OCR_AVAILABLE = None


def is_ocr_available() -> bool:
    """True if Tesseract binary is reachable (via PATH, TESSERACT_CMD, or Windows defaults)."""
    global _OCR_AVAILABLE
    if _OCR_AVAILABLE is not None:
        return _OCR_AVAILABLE
    try:
        import pytesseract
    except ImportError as exc:
        _OCR_AVAILABLE = False
        logger.warning(
            "OCR disabled — Python packages missing for this interpreter (%s). "
            "Run: python3 -m pip install pytesseract Pillow — %s",
            sys.executable,
            exc,
        )
        return _OCR_AVAILABLE

    cmd = _configure_tesseract()
    if not cmd:
        _OCR_AVAILABLE = False
        logger.warning(
            "OCR disabled — pytesseract is installed but the Tesseract OCR program was not found. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki (add to PATH), or set in "
            "web-ui/.env: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe "
            "(python: %s)",
            sys.executable,
        )
        return _OCR_AVAILABLE

    try:
        version = pytesseract.get_tesseract_version()
        _OCR_AVAILABLE = True
        logger.info("OCR enabled (Tesseract %s at %s)", version, cmd)
    except Exception as exc:
        _OCR_AVAILABLE = False
        logger.warning(
            "OCR disabled — Tesseract at %s failed: %s",
            cmd,
            exc,
        )
    return _OCR_AVAILABLE


def _decode_screenshot(b64_data: str):
    from PIL import Image

    raw = base64.b64decode(b64_data)
    image = Image.open(io.BytesIO(raw))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _parse_ocr_data(data: Dict) -> List[OcrLine]:
    lines: List[OcrLine] = []
    n = len(data.get("text", []))
    grouped: Dict[Tuple[int, int, int], List[dict]] = {}

    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = 0.0
        if conf < _MIN_CONFIDENCE:
            continue
        key = (
            int(data.get("block_num", [0])[i]),
            int(data.get("par_num", [0])[i]),
            int(data.get("line_num", [0])[i]),
        )
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "conf": conf,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )

    for parts in grouped.values():
        text = " ".join(p["text"] for p in parts).strip()
        if len(text) < 2:
            continue
        left = min(p["left"] for p in parts)
        top = min(p["top"] for p in parts)
        right = max(p["left"] + p["width"] for p in parts)
        bottom = max(p["top"] + p["height"] for p in parts)
        conf = sum(p["conf"] for p in parts) / len(parts)
        lines.append(
            OcrLine(
                text=text,
                confidence=conf,
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
            )
        )

    lines.sort(key=lambda ln: (ln.top, ln.left))
    return lines


def _run_ocr_on_image(image) -> List[OcrLine]:
    import pytesseract

    _configure_tesseract()
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return _parse_ocr_data(data)


def ocr_lines_as_text_list(lines: List[OcrLine]) -> List[str]:
    return [ln.text for ln in lines]


def _format_ocr_lines(lines: List[OcrLine], *, url: str = "") -> str:
    if not lines:
        return "No readable text detected on screenshot."

    header = "Visible text from screenshot (OCR — use with DOM indexes, not instead of them):"
    if url:
        header += f"\nPage: {url[:200]}"

    body_lines: List[str] = []
    highlights: List[str] = []
    for ln in lines:
        body_lines.append(ln.text)
        if _FORM_KEYWORDS.search(ln.text) and ln.text not in highlights:
            highlights.append(ln.text)

    body = "\n".join(body_lines)
    if len(body) > _MAX_OCR_CHARS:
        body = body[:_MAX_OCR_CHARS] + "\n…(OCR truncated)"

    out = header + "\n\n" + body
    if highlights:
        out += "\n\n[OCR highlights — form/search related]\n- " + "\n- ".join(highlights[:25])

    advance = [ln.text for ln in lines if _FORM_KEYWORDS.search(ln.text) and
               re.search(r"\b(next|continue|submit|review|apply|save)\b", ln.text, re.I)]
    if advance:
        out += "\n\n[OCR — advance when section complete]\n- " + "\n- ".join(advance[:10])
    return out


async def get_ocr_lines_from_screenshot_b64(b64_data: str) -> List[OcrLine]:
    if not b64_data or not is_ocr_available():
        return []

    key = _screenshot_cache_key(b64_data)
    cached = _ocr_line_cache.get(key)
    if cached is not None:
        return cached

    def _work() -> List[OcrLine]:
        try:
            image = _decode_screenshot(b64_data)
            return _run_ocr_on_image(image)
        except Exception:
            return []

    lines = await asyncio.to_thread(_work)
    return _cache_ocr_lines(key, lines)


def extract_text_from_image_b64(b64_data: str) -> Optional[str]:
    """Synchronous OCR pipeline; returns formatted string for the LLM."""
    if not b64_data or not is_ocr_available():
        return None
    try:
        image = _decode_screenshot(b64_data)
        lines = _run_ocr_on_image(image)
        return _format_ocr_lines(lines)
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return None


async def extract_text_from_screenshot_b64(
    b64_data: str,
    *,
    url: str = "",
) -> Optional[str]:
    if not b64_data:
        return None
    if not is_ocr_available():
        return None

    def _work() -> Optional[str]:
        try:
            image = _decode_screenshot(b64_data)
            lines = _run_ocr_on_image(image)
            return _format_ocr_lines(lines, url=url)
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return None

    return await asyncio.to_thread(_work)


async def extract_text_from_page(page, *, url: str = "") -> Optional[str]:
    """Capture viewport screenshot and OCR it."""
    try:
        png_bytes = await page.screenshot(type="png", full_page=False)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return await extract_text_from_screenshot_b64(b64, url=url)
    except Exception as exc:
        logger.debug("page screenshot for OCR failed: %s", exc)
        return None


def find_text_matches(lines: List[OcrLine], query: str, limit: int = 10) -> List[OcrLine]:
    q = query.strip().lower()
    if not q:
        return []
    matches: List[OcrLine] = []
    for ln in lines:
        if q in ln.text.lower():
            matches.append(ln)
    matches.sort(key=lambda ln: (-len(ln.text), ln.top))
    return matches[:limit]


async def search_visible_text(b64_data: str, query: str) -> str:
    """Find OCR lines matching query (for search/forms when DOM misses labels)."""
    if not b64_data or not is_ocr_available():
        return "OCR not available."

    def _work() -> str:
        try:
            image = _decode_screenshot(b64_data)
            lines = _run_ocr_on_image(image)
            hits = find_text_matches(lines, query)
            if not hits:
                return f"OCR: no visible text matching '{query}'."
            parts = [
                f'"{h.text}" (~y={h.vertical_center}, conf={h.confidence:.0f})'
                for h in hits
            ]
            return "OCR matches for '" + query + "':\n- " + "\n- ".join(parts)
        except Exception as exc:
            return f"OCR search failed: {exc}"

    return await asyncio.to_thread(_work)
