from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies (ported from the chat/ project).
# These are imported lazily so Friday still runs without them installed.
# ---------------------------------------------------------------------------

try:  # PyMuPDF for PDF text extraction
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:  # SerpAPI for structured web search (chat/ used google-search-results)
    from serpapi import GoogleSearch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GoogleSearch = None

try:  # OpenRouter fallback client (chat/ used the openai SDK)
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


# ---------------------------------------------------------------------------
# PDF text extraction (from chat/ extract_text_from_pdf)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_data: bytes, max_chars: int = 8000) -> str:
    """Extract text content from PDF file data using PyMuPDF.

    Returns the extracted text, or an error/warning string prefixed with
    'Error:' / 'Warning:' so callers can surface it to the user.
    """
    if fitz is None:
        return "Error: PyMuPDF (fitz) is not installed. Install with: pip install PyMuPDF"
    if not pdf_data:
        return "Error: No PDF data provided"
    try:
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        if not pdf_document:
            return "Error: Could not open PDF document"
        if pdf_document.page_count == 0:
            pdf_document.close()
            return "Error: PDF document has no pages"

        text_content = ""
        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document[page_num]
                if page:
                    page_text = page.get_text()
                    if page_text:
                        text_content += f"Page {page_num + 1}:\n{page_text}\n\n"
            except Exception as page_error:
                logger.warning(f"Error extracting text from page {page_num + 1}: {page_error}")
                text_content += f"Page {page_num + 1}: [Error extracting text]\n\n"

        pdf_document.close()
        final_text = text_content.strip()
        if not final_text:
            return "Warning: PDF appears to be empty or contains no extractable text (might be image-based PDF)"
        if len(final_text) > max_chars:
            final_text = (
                final_text[:max_chars]
                + "\n\n[Note: PDF content truncated due to length. Only the first portion is shown.]"
            )
        return final_text
    except Exception as exc:
        logger.error(f"Error extracting text from PDF: {exc}")
        return f"Error reading PDF: {exc}"


# ---------------------------------------------------------------------------
# SerpAPI structured web search (from chat/ perform_search / format_search_results)
# ---------------------------------------------------------------------------

def serpapi_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web using SerpAPI. Returns a dict with results or an error."""
    if GoogleSearch is None:
        return {"error": "SerpAPI client (google-search-results) is not installed. Install with: pip install google-search-results"}
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        return {"error": "SERPAPI_KEY is not set in environment variables."}
    try:
        search = GoogleSearch({"q": query, "api_key": api_key})
        results = search.get_dict()
        organic = results.get("organic_results", [])
        formatted = "Here are the top search results:\n\n"
        for result in organic[:max_results]:
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No snippet")
            formatted += f"**{title}**\n{snippet}\n\n"
        return {
            "query": query,
            "source": "serpapi",
            "formatted": formatted.strip() if organic else "No search results found.",
            "organic_results": organic[:max_results],
        }
    except Exception as exc:
        logger.error(f"SerpAPI search failed: {exc}")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# OpenRouter fallback client (from chat/ OpenRouter usage)
# ---------------------------------------------------------------------------

def _openrouter_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def openrouter_chat(
    messages: List[Dict[str, Any]],
    model: str = "openai/gpt-4o-mini",
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 45,
) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenRouter as a fallback LLM. Returns (content, model_used) or (None, None)."""
    client = _openrouter_client()
    if client is None:
        return None, None
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost"),
                "X-Title": "Friday Agent",
            },
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if completion and completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content
            if content and len(content.strip()) > 5:
                return content.strip(), model
    except Exception as exc:
        logger.warning(f"OpenRouter fallback failed: {exc}")
    return None, None


def openrouter_vision(
    text: str,
    image_b64: str,
    mime_type: str,
    model: str = "openai/gpt-4o-mini",
    *,
    max_tokens: int = 4096,
    timeout: int = 45,
) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenRouter vision model with an inline base64 image. Returns (content, model_used)."""
    client = _openrouter_client()
    if client is None:
        return None, None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }
    ]
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost"),
                "X-Title": "Friday Agent",
            },
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if completion and completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content
            if content and len(content.strip()) > 5:
                return content.strip(), model
    except Exception as exc:
        logger.warning(f"OpenRouter vision fallback failed: {exc}")
    return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_file_base64(path: Path) -> Tuple[str, str]:
    """Read a file and return (base64_str, mime_type)."""
    from friday.runtime.files import guess_mime

    data = path.read_bytes()
    mime = guess_mime(path)
    return base64.b64encode(data).decode("utf-8"), mime


def read_pdf_text(path: Path, max_chars: int = 8000) -> str:
    """Convenience wrapper: extract text from a PDF file on disk."""
    return extract_text_from_pdf(path.read_bytes(), max_chars=max_chars)
