"""
Unified page understanding for job applications (LinkedIn and similar).

Combines DOM probes, scroll metrics, form progress, OCR, and validation errors
into one briefing the agent receives every step.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.browser.scroll_helpers import get_remaining_scroll, scroll_to_edge, format_scroll_result

if TYPE_CHECKING:
    from browser_use.browser.context import BrowserContext
    from browser_use.browser.views import BrowserState

logger = logging.getLogger(__name__)

_ERRORS_JS = r"""
() => {
  const errors = [];
  const seen = new Set();
  const add = (t) => {
    const s = (t || '').replace(/\s+/g, ' ').trim();
    if (s.length < 4 || s.length > 300 || seen.has(s)) return;
    seen.add(s);
    errors.push(s);
  };
  const selectors = [
    '[role="alert"]', '[aria-invalid="true"]', '.artdeco-inline-feedback--error',
    '[class*="error" i]', '[class*="invalid" i]', '[data-test*="error" i]',
  ];
  for (const sel of selectors) {
    document.querySelectorAll(sel).forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) add(el.innerText || el.textContent);
    });
  }
  return errors.slice(0, 12);
}
"""

_BUTTONS_JS = r"""
() => {
  const buttons = [];
  const seen = new Set();
  const add = (text, tag, role) => {
    const t = (text || '').replace(/\s+/g, ' ').trim();
    if (t.length < 2 || t.length > 80 || seen.has(t)) return;
    seen.add(t);
    buttons.push({ text: t, tag, role: role || '' });
  };
  for (const el of document.querySelectorAll('button, [role="button"], a[role="button"]')) {
    const r = el.getBoundingClientRect();
    if (r.width < 20 || r.height < 10 || r.bottom < 0 || r.top > window.innerHeight) continue;
    add(el.innerText || el.getAttribute('aria-label'), el.tagName, el.getAttribute('role'));
  }
  return buttons.slice(0, 30);
}
"""

_LINKEDIN_HINT_RE = re.compile(
    r"linkedin\.com|easy\s*apply|application\s+modal|review\s+application|"
    r"contact\s+info|additional\s+questions|resume",
    re.IGNORECASE,
)
_SUBMIT_RE = re.compile(
    r"\b(submit|submit application|review|next|continue|done|apply)\b",
    re.IGNORECASE,
)


async def probe_validation_errors(page) -> List[str]:
    try:
        return await page.evaluate(_ERRORS_JS) or []
    except Exception:
        return []


async def probe_visible_buttons(page) -> List[Dict[str, str]]:
    try:
        return await page.evaluate(_BUTTONS_JS) or []
    except Exception:
        return []


def _match_buttons_to_indexes(elements_text: str, button_texts: List[str]) -> List[str]:
    """Map visible button labels to DOM indexes when possible."""
    mapped: List[str] = []
    lines = elements_text.split("\n") if elements_text else []
    for btn in button_texts:
        t = btn.get("text", "")
        if not t or not _SUBMIT_RE.search(t):
            continue
        for line in lines:
            if t.lower() in line.lower() or (len(t) > 4 and t.lower()[:12] in line.lower()):
                m = re.search(r"\[(\d+)\]", line)
                if m:
                    mapped.append(f"[{m.group(1)}] {t}")
                    break
    return mapped[:12]


async def discover_footer_controls(browser: "BrowserContext") -> str:
    """
    Scroll toward the bottom in small steps (not one jump), then report buttons.
    """
    page = await browser.get_current_page()
    result = await scroll_to_edge(page, "bottom")
    await asyncio.sleep(0.1)
    remaining = await get_remaining_scroll(page)
    buttons = await probe_visible_buttons(page)
    footer_btns = [b["text"] for b in buttons if _SUBMIT_RE.search(b.get("text", ""))]

    parts = [
        "[Footer discovery] Scrolled toward bottom in small steps (centered popup/page).",
        format_scroll_result(result, "Scroll"),
    ]
    rb = int(remaining.get("below") or 0)
    if rb > 40:
        parts.append(f"Still ~{rb}px below — run reveal_more_content or scroll_down again.")
    if footer_btns:
        parts.append("Buttons near bottom (OCR/DOM): " + ", ".join(footer_btns[:10]))
    return "\n".join(parts)


async def build_full_page_briefing(
    browser: "BrowserContext",
    state: "BrowserState",
    *,
    include_footer_scan: bool = False,
    include_html_inspect: bool = True,
    deep_html: bool = False,
    enable_ocr: bool = True,
    previous_fingerprint: Optional[str] = None,
    skip_unchanged_full: bool = True,
) -> Optional[str]:
    """
    Maximum page understanding via parallel scrape pipeline (DOM, UI, form, OCR).
    """
    from src.browser.page_scrape_pipeline import build_briefing_from_scrape

    briefing, _fp = await build_briefing_from_scrape(
        browser,
        state,
        include_footer_scan=include_footer_scan,
        include_html_inspect=include_html_inspect,
        deep_html=deep_html,
        enable_ocr=enable_ocr,
        job_application_mode=True,
        previous_fingerprint=previous_fingerprint,
        skip_unchanged_full=skip_unchanged_full,
    )
    return briefing


async def run_full_page_scan(browser: "BrowserContext") -> str:
    """Agent-callable full scan (scroll bottom + refresh understanding)."""
    state = await browser.get_state(cache_clickable_elements_hashes=False)
    briefing = await build_full_page_briefing(
        browser,
        state,
        include_footer_scan=True,
        include_html_inspect=True,
        deep_html=True,
    )
    return briefing or "Page scan completed; no extra signals."
