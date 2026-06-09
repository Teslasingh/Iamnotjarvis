"""
Unified page scrape → parse → LLM briefing pipeline.

Runs DOM, form, UI, validation, button, and OCR probes in parallel, builds one
structured snapshot, and formats a single briefing for the agent each step.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.browser.form_progress import build_form_progress_message, probe_form_fields
from src.browser.html_inspector import (
    _DEEP_MAX_CHARS,
    _MAX_CHARS,
    extract_readable_html_text,
    format_dom_inspection,
    probe_dom_structure,
)
from src.browser.ocr_service import (
    extract_text_from_page,
    extract_text_from_screenshot_b64,
    get_ocr_lines_from_screenshot_b64,
    is_ocr_available,
    ocr_lines_as_text_list,
)
from src.browser.page_understanding import (
    _LINKEDIN_HINT_RE,
    _SUBMIT_RE,
    _match_buttons_to_indexes,
    discover_footer_controls,
    probe_validation_errors,
    probe_visible_buttons,
)
from src.browser.search_scrape import url_looks_like_search_results
from src.browser.ui_context import build_ui_context_hints

if TYPE_CHECKING:
    from browser_use.browser.context import BrowserContext
    from browser_use.browser.views import BrowserState

logger = logging.getLogger(__name__)

PARALLEL_SCRAPE = os.getenv("PAGE_SCRAPE_PARALLEL", "true").lower() in (
    "1", "true", "yes", "on"
)
FAST_SCRAPE = os.getenv("PAGE_SCRAPE_FAST", "true").lower() in (
    "1", "true", "yes", "on"
)
OCR_STEP_INTERVAL = max(1, int(os.getenv("OCR_STEP_INTERVAL", "4")))
FAST_HTML_MAX_CHARS = int(os.getenv("HTML_INSPECT_FAST_MAX_CHARS", "2800"))


@dataclass
class PageSnapshot:
    """Parsed page state fed to the LLM in one briefing block."""

    fingerprint: str
    url: str
    title: str
    dom_data: Dict[str, Any] = field(default_factory=dict)
    html_brief: str = ""
    ui_hints: str = ""
    form_probe: Dict[str, Any] = field(default_factory=dict)
    form_progress: str = ""
    validation_errors: List[str] = field(default_factory=list)
    visible_buttons: List[Dict[str, str]] = field(default_factory=list)
    nav_mapped: List[str] = field(default_factory=list)
    ocr_lines: List[Any] = field(default_factory=list)
    ocr_text: str = ""
    footer_scan: str = ""
    pixels_above: int = 0
    pixels_below: int = 0
    changed_from_previous: bool = True


def compute_page_fingerprint(
    url: str,
    title: str,
    state: "BrowserState",
    form_probe: Dict[str, Any],
    dom_data: Dict[str, Any],
) -> str:
    """Stable hash for skip-if-unchanged briefing optimization."""
    parts: List[str] = [url[:200], title[:80]]
    try:
        sm = getattr(state, "selector_map", None) or {}
        hashes = sorted(
            str(getattr(e.hash, "branch_path_hash", e))
            for e in (sm.values() if hasattr(sm, "values") else [])
        )[:80]
        parts.append("|".join(hashes))
    except Exception:
        try:
            parts.append(
                state.element_tree.clickable_elements_to_string()[:1200]
            )
        except Exception:
            pass
    req = form_probe.get("required_empty") or []
    parts.append(json.dumps(req, sort_keys=True, default=str)[:500])
    meta = dom_data.get("meta") or {}
    parts.append(
        f"{meta.get('dialogCount', 0)}:{meta.get('visibleFieldCount', 0)}"
    )
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


async def _run_probe(name: str, coro):
    try:
        return await coro
    except Exception as exc:
        logger.debug("Probe %s failed: %s", name, exc)
        return None


async def quick_page_fingerprint(
    browser: "BrowserContext",
    state: "BrowserState",
) -> str:
    """Fast fingerprint (form + indexes only) to skip heavy scrape when unchanged."""
    page = await browser.get_current_page()
    url = getattr(state, "url", "") or ""
    title = getattr(state, "title", "") or ""
    form_probe = await probe_form_fields(page) or {}
    return compute_page_fingerprint(url, title, state, form_probe, {})


def _should_run_ocr(
    *,
    enable_ocr: bool,
    step_number: int,
    force_ocr: bool,
    screenshot: Optional[str],
    use_vision: bool,
) -> bool:
    if not enable_ocr or not is_ocr_available():
        return False
    if not screenshot:
        return False
    if force_ocr or step_number <= 1:
        return True
    if not FAST_SCRAPE:
        return True
    if use_vision:
        return False
    return step_number % OCR_STEP_INTERVAL == 0


async def scrape_page_parallel(
    browser: "BrowserContext",
    state: "BrowserState",
    *,
    include_html_inspect: bool = True,
    deep_html: bool = False,
    include_footer_scan: bool = False,
    enable_ocr: bool = True,
    previous_fingerprint: Optional[str] = None,
    fast: bool = True,
    step_number: int = 0,
    force_ocr: bool = False,
    use_vision: bool = False,
) -> PageSnapshot:
    """
    Scrape the live page with all available tools in parallel, then parse.
    """
    page = await browser.get_current_page()
    url = getattr(state, "url", "") or ""
    title = getattr(state, "title", "") or ""
    pixels_above = int(getattr(state, "pixels_above", 0) or 0)
    pixels_below = int(getattr(state, "pixels_below", 0) or 0)
    screenshot = getattr(state, "screenshot", None)

    dom_data: Dict[str, Any] = {}
    ui_hints = ""
    form_probe: Dict[str, Any] = {}
    errors: List[str] = []
    buttons: List[Dict[str, str]] = []
    ocr_lines: List = []
    readable_text: Optional[str] = None

    ocr_text = ""

    run_ocr = _should_run_ocr(
        enable_ocr=enable_ocr,
        step_number=step_number,
        force_ocr=force_ocr,
        screenshot=screenshot,
        use_vision=use_vision,
    )
    use_fast = fast and FAST_SCRAPE

    if PARALLEL_SCRAPE:
        probes: Dict[str, Any] = {
            "form": probe_form_fields(page),
            "errors": probe_validation_errors(page),
        }
        if not use_fast:
            probes["dom"] = probe_dom_structure(page)
            probes["ui"] = build_ui_context_hints(state, browser)
            probes["buttons"] = probe_visible_buttons(page)
        else:
            probes["ui"] = build_ui_context_hints(state, browser)
        if include_html_inspect and not use_fast:
            probes["readable"] = extract_readable_html_text(page)
        elif include_html_inspect:
            probes["dom"] = probe_dom_structure(page)
        if run_ocr and screenshot:
            probes["ocr"] = get_ocr_lines_from_screenshot_b64(screenshot)
        elif run_ocr and not screenshot:
            probes["ocr"] = extract_text_from_page(page, url=url)

        names = list(probes.keys())
        raw_results = await asyncio.gather(
            *[_run_probe(n, probes[n]) for n in names]
        )
        parsed = dict(zip(names, raw_results))

        if isinstance(parsed.get("dom"), dict):
            dom_data = parsed["dom"]
        if isinstance(parsed.get("ui"), str):
            ui_hints = parsed["ui"]
        if isinstance(parsed.get("form"), dict):
            form_probe = parsed["form"]
        if isinstance(parsed.get("errors"), list):
            errors = parsed["errors"]
        if isinstance(parsed.get("buttons"), list):
            buttons = parsed["buttons"]
        elif use_fast:
            buttons = await probe_visible_buttons(page) or []
        if isinstance(parsed.get("readable"), str):
            readable_text = parsed["readable"]
        ocr_raw = parsed.get("ocr")
        if isinstance(ocr_raw, list):
            ocr_lines = ocr_raw
        elif isinstance(ocr_raw, str):
            ocr_text = ocr_raw
    else:
        dom_data = await probe_dom_structure(page) or {}
        ui_hints = await build_ui_context_hints(state, browser) or ""
        form_probe = await probe_form_fields(page) or {}
        errors = await probe_validation_errors(page) or []
        buttons = await probe_visible_buttons(page) or []
        if include_html_inspect:
            readable_text = await extract_readable_html_text(page)
        if enable_ocr and is_ocr_available():
            if screenshot:
                ocr_lines = await get_ocr_lines_from_screenshot_b64(screenshot) or []
            else:
                ocr_text = await extract_text_from_page(page, url=url) or ""

    html_brief = ""
    if include_html_inspect and dom_data:
        if deep_html:
            max_c = _DEEP_MAX_CHARS
        elif use_fast:
            max_c = FAST_HTML_MAX_CHARS
        else:
            max_c = _MAX_CHARS
        html_brief = format_dom_inspection(dom_data, max_chars=max_c)
        if readable_text and not use_fast:
            html_brief += (
                "\n\n[Readable page text from HTML extract]\n"
                + readable_text[:2000]
            )

    if ocr_lines and not ocr_text:
        ocr_text = "\n".join(ocr_lines_as_text_list(ocr_lines)[:40 if use_fast else 60])

    progress = build_form_progress_message(
        state,
        dom_probe=form_probe,
        ocr_line_texts=ocr_lines_as_text_list(ocr_lines) if ocr_lines else [],
    )

    elements_text = ""
    try:
        elements_text = state.element_tree.clickable_elements_to_string()
    except Exception:
        pass
    nav_mapped = _match_buttons_to_indexes(elements_text, buttons)

    fingerprint = compute_page_fingerprint(url, title, state, form_probe, dom_data)
    changed = previous_fingerprint is None or fingerprint != previous_fingerprint

    footer_scan = ""
    if include_footer_scan:
        footer_scan = await discover_footer_controls(browser)

    return PageSnapshot(
        fingerprint=fingerprint,
        url=url,
        title=title,
        dom_data=dom_data,
        html_brief=html_brief,
        ui_hints=ui_hints or "",
        form_probe=form_probe,
        form_progress=progress or "",
        validation_errors=errors,
        visible_buttons=buttons,
        nav_mapped=nav_mapped,
        ocr_lines=ocr_lines,
        ocr_text=ocr_text or "",
        footer_scan=footer_scan,
        pixels_above=pixels_above,
        pixels_below=pixels_below,
        changed_from_previous=changed,
    )


def format_unified_briefing(
    snapshot: PageSnapshot,
    *,
    job_application_mode: bool = True,
    minimal_if_unchanged: bool = False,
) -> str:
    """Single LLM-ready briefing from a PageSnapshot."""
    if minimal_if_unchanged and not snapshot.changed_from_previous:
        return (
            "[Page unchanged — use current element indexes]\n"
            f"URL: {snapshot.url[:180]}"
        )

    sections: List[str] = [
        "[Page briefing]",
        f"URL: {snapshot.url[:220]}\nTitle: {snapshot.title[:120]}",
    ]

    if job_application_mode and _LINKEDIN_HINT_RE.search(
        snapshot.url + " " + snapshot.title
    ):
        sections.append(
            "[Job application mode]\n"
            "- Fill required empty only; skip fields already filled in Form progress.\n"
            "- Dropdowns: get_dropdown_options → select_dropdown_option.\n"
            "- Scroll: small scroll_down steps; scroll_element_into_view for a field index.\n"
            "- If top fields were missed: scroll_to_top or scroll_up.\n"
            "- Review/submit only: discover_footer_controls or scroll_to_bottom."
        )

    above = snapshot.pixels_above
    if above >= 120:
        sections.append(
            f"[Scroll] ~{above}px above viewport — scroll_up / scroll_to_top before scrolling down."
        )
    elif snapshot.pixels_below >= 200:
        sections.append(
            f"[Scroll] ~{snapshot.pixels_below}px below — scroll_down in steps; "
            "do not jump to bottom until review/submit."
        )

    if url_looks_like_search_results(snapshot.url):
        sections.append(
            "[Search results page]\n"
            "- Read [OCR visible text] and element indexes for job titles and Apply links.\n"
            "- Use ocr_search_visible_text if a label or button is missing from the DOM list.\n"
            "- Wait for listings to finish loading before clicking; scroll_down to load more cards if needed."
        )

    if snapshot.html_brief:
        sections.append(snapshot.html_brief)
    if snapshot.ui_hints:
        sections.append(snapshot.ui_hints)
    if snapshot.form_progress:
        sections.append(snapshot.form_progress)
    if snapshot.validation_errors:
        sections.append(
            "[Validation errors — fix before Next/Submit]\n- "
            + "\n- ".join(snapshot.validation_errors)
        )

    ocr_submit: List[str] = []
    if snapshot.ocr_text:
        ocr_submit = [
            t
            for t in snapshot.ocr_text.split("\n")
            if _SUBMIT_RE.search(t)
        ]
    if snapshot.nav_mapped or ocr_submit:
        sections.append("[Navigation targets]")
        if snapshot.nav_mapped:
            sections.append("DOM indexes: " + ", ".join(snapshot.nav_mapped))
        if ocr_submit:
            sections.append("OCR labels: " + ", ".join(ocr_submit[:8]))

    if snapshot.ocr_text:
        cap = 2000 if FAST_SCRAPE else 3500
        sections.append(
            "[OCR visible text]\n" + snapshot.ocr_text[:cap]
        )

    if snapshot.footer_scan:
        sections.append(snapshot.footer_scan)

    return "\n\n".join(sections)


async def build_briefing_from_scrape(
    browser: "BrowserContext",
    state: "BrowserState",
    *,
    include_footer_scan: bool = False,
    include_html_inspect: bool = True,
    deep_html: bool = False,
    enable_ocr: bool = True,
    job_application_mode: bool = True,
    previous_fingerprint: Optional[str] = None,
    skip_unchanged_full: bool = True,
    fast: bool = True,
    step_number: int = 0,
    force_ocr: bool = False,
    use_vision: bool = False,
    force_full_scrape: bool = False,
) -> tuple[Optional[str], str]:
    """
    Run scrape pipeline and return (briefing_text, new_fingerprint).
    Skips heavy probes when the quick fingerprint matches the previous step.
    """
    url = getattr(state, "url", "") or ""
    if force_full_scrape or url_looks_like_search_results(url):
        skip_unchanged_full = False
        fast = False
        force_ocr = True

    quick_fp = await quick_page_fingerprint(browser, state)
    if (
        skip_unchanged_full
        and FAST_SCRAPE
        and previous_fingerprint
        and quick_fp == previous_fingerprint
    ):
        snap = PageSnapshot(
            fingerprint=quick_fp,
            url=getattr(state, "url", "") or "",
            title=getattr(state, "title", "") or "",
            changed_from_previous=False,
        )
        return (
            format_unified_briefing(
                snap,
                job_application_mode=job_application_mode,
                minimal_if_unchanged=True,
            ),
            quick_fp,
        )

    snapshot = await scrape_page_parallel(
        browser,
        state,
        include_html_inspect=include_html_inspect,
        deep_html=deep_html,
        include_footer_scan=include_footer_scan,
        enable_ocr=enable_ocr,
        previous_fingerprint=previous_fingerprint,
        fast=fast and not deep_html,
        step_number=step_number,
        force_ocr=force_ocr,
        use_vision=use_vision,
    )
    minimal = skip_unchanged_full and not snapshot.changed_from_previous
    text = format_unified_briefing(
        snapshot,
        job_application_mode=job_application_mode,
        minimal_if_unchanged=minimal,
    )
    return text or None, snapshot.fingerprint


# Actions that typically change DOM / new options — trigger fast re-scrape
DOM_CHANGING_ACTIONS = frozenset({
    "click_element_by_index",
    "input_text",
    "select_dropdown_option",
    "get_dropdown_options",
    "scroll_to_bottom",
    "reveal_more_content",
    "advance_form_step",
    "discover_footer_controls",
    "scan_application_page",
    "go_to_url",
    "open_tab",
    "switch_tab",
    "send_keys",
})
