"""Per-step UI context analysis for goal-driven browser automation."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from browser_use.browser.views import BrowserState

logger = logging.getLogger(__name__)

_NAV_RE = re.compile(
    r"\b(next|continue|submit|apply|save|proceed|confirm|finish|done|send)\b",
    re.IGNORECASE,
)
_MODAL_ROLE_RE = re.compile(r"role=['\"]?(dialog|alertdialog)['\"]?", re.IGNORECASE)
_DIALOG_TEXT_RE = re.compile(
    r"<(dialog|div|section)[^>]*>|modal|drawer|wizard|step\s+\d|application",
    re.IGNORECASE,
)


async def _page_ui_signals(page) -> List[str]:
    """Lightweight DOM probes via Playwright (no rigid site scripts)."""
    signals: List[str] = []
    try:
        probe = await page.evaluate(
            """() => {
                const out = {
                    dialogs: 0, modals: 0, scrollables: 0, fixedOverlays: 0,
                    scrollTarget: null, remainingBelow: 0, remainingAbove: 0
                };
                const scrollCandidates = [];
                const considerScroll = (el, priority) => {
                    if (!el || el === document.body) return;
                    const r = el.getBoundingClientRect();
                    if (r.width < 80 || r.height < 80) return;
                    const st = getComputedStyle(el);
                    const canScroll = el.scrollHeight > el.clientHeight + 24;
                    if (canScroll && (st.overflowY === 'auto' || st.overflowY === 'scroll' || priority >= 5)) {
                        scrollCandidates.push({
                            priority, area: r.width * r.height,
                            below: el.scrollHeight - el.clientHeight - el.scrollTop,
                            above: el.scrollTop,
                            role: el.getAttribute('role') || el.tagName
                        });
                        out.scrollables += 1;
                    }
                };
                for (const el of document.querySelectorAll('[role="dialog"], [role="alertdialog"], dialog[open]')) {
                    const r = el.getBoundingClientRect();
                    if (r.width > 50 && r.height > 50) out.dialogs += 1;
                    considerScroll(el, 10);
                    for (const c of el.querySelectorAll('*')) considerScroll(c, 8);
                }
                for (const el of document.querySelectorAll('[aria-modal="true"], .modal, [class*="modal" i]')) {
                    const r = el.getBoundingClientRect();
                    if (r.width > 80 && r.height > 80) out.modals += 1;
                    considerScroll(el, 6);
                }
                scrollCandidates.sort((a, b) => b.priority - a.priority || b.area - a.area);
                if (scrollCandidates.length) {
                    const best = scrollCandidates[0];
                    out.scrollTarget = best.role;
                    out.remainingBelow = Math.round(best.below);
                    out.remainingAbove = Math.round(best.above);
                } else {
                    out.remainingBelow = Math.max(0, document.documentElement.scrollHeight - window.innerHeight - window.scrollY);
                    out.remainingAbove = Math.round(window.scrollY);
                    if (out.remainingBelow > 40 || out.remainingAbove > 40) out.scrollTarget = 'window';
                }
                for (const el of document.querySelectorAll('*')) {
                    const st = getComputedStyle(el);
                    if (st.position === 'fixed' && parseInt(st.zIndex || '0', 10) > 100) {
                        const r = el.getBoundingClientRect();
                        if (r.width > window.innerWidth * 0.3 && r.height > window.innerHeight * 0.2) {
                            out.fixedOverlays += 1;
                            if (out.fixedOverlays > 3) break;
                        }
                    }
                }
                return out;
            }"""
        )
        if probe.get("dialogs", 0) > 0:
            signals.append(
                f"Detected {probe['dialogs']} dialog/alertdialog region(s)—scroll_down scrolls from viewport center into that popup."
            )
        if probe.get("modals", 0) > 0:
            signals.append(
                f"Detected {probe['modals']} modal-like overlay(s)—scroll inside the panel; window-only scroll may not move content."
            )
        remaining_below = int(probe.get("remainingBelow") or 0)
        remaining_above = int(probe.get("remainingAbove") or 0)
        scroll_target = probe.get("scrollTarget")
        if remaining_above > 100:
            signals.append(
                f"~{remaining_above}px content above—use scroll_up or scroll_to_top before scrolling down."
            )
        if remaining_below > 200 and remaining_above < 100:
            where = f"inside {scroll_target}" if scroll_target else "on page"
            signals.append(
                f"~{remaining_below}px more content below {where}—use scroll_down in small steps or scroll_element_into_view."
            )
        if probe.get("scrollables", 0) > 0 and remaining_below <= 200:
            signals.append(
                f"Found {probe['scrollables']} scrollable region(s)—if a control is missing, scroll_element_into_view on its index."
            )
        if probe.get("fixedOverlays", 0) > 0:
            signals.append(
                "Large fixed-position overlay(s) present—may block clicks until closed or scrolled past."
            )
    except Exception as exc:
        logger.debug("page UI probe failed: %s", exc)
    return signals


def _element_tree_hints(elements_text: str, state: "BrowserState") -> List[str]:
    hints: List[str] = []
    if not elements_text or elements_text.strip() == "empty page":
        hints.append(
            "No interactive elements in the current viewport—use scroll_down or scroll_to_bottom "
            "(scrolls modal/panel first), then scroll_to_text if you know a label."
        )
        return hints

    lines = elements_text.split("\n")
    nav_indices: List[str] = []
    file_inputs: List[str] = []
    for line in lines:
        m = re.search(r"\[(\d+)\]", line)
        if not m:
            continue
        idx = m.group(1)
        if _NAV_RE.search(line):
            nav_indices.append(f"[{idx}]")
        if re.search(r"type=['\"]?file|<input|upload", line, re.I):
            file_inputs.append(f"[{idx}]")

    if _MODAL_ROLE_RE.search(elements_text) or _DIALOG_TEXT_RE.search(elements_text):
        hints.append(
            "Element tree suggests a multi-step or modal UI—treat each section as part of one workflow; scroll before concluding controls are absent."
        )
    if nav_indices:
        hints.append(
            "Possible workflow controls: " + ", ".join(nav_indices[:8])
            + (" …" if len(nav_indices) > 8 else "")
            + " — verify the active step changed after clicking."
        )
    if file_inputs:
        hints.append(
            "File upload field(s) at " + ", ".join(file_inputs[:4]) + " — use upload_file when a path is available."
        )
    if re.search(r"combobox|listbox|select|dropdown", elements_text, re.I):
        hints.append(
            "Dropdown/combobox detected — use get_dropdown_options then select_dropdown_option (fuzzy text match)."
        )

    above = getattr(state, "pixels_above", 0) or 0
    below = getattr(state, "pixels_below", 0) or 0
    if above > 120:
        hints.append(
            f"~{above}px content above viewport—use scroll_up or scroll_to_top before scrolling down."
        )
    if below > 220 and above < 100:
        hints.append(
            f"~{below}px below viewport—use scroll_down in small steps or scroll_element_into_view; "
            "avoid scroll_to_bottom until review/submit."
        )

    return hints


async def build_ui_context_hints(state: "BrowserState", browser_context) -> Optional[str]:
    """
    Build a short, state-specific hint block for the LLM (not a fixed script).
    """
    try:
        elements_text = state.element_tree.clickable_elements_to_string()
    except Exception:
        elements_text = ""

    parts: List[str] = []
    parts.extend(_element_tree_hints(elements_text, state))

    try:
        page = await browser_context.get_current_page()
        parts.extend(await _page_ui_signals(page))
    except Exception as exc:
        logger.debug("UI page signals skipped: %s", exc)

    if not parts:
        return None

    title = getattr(state, "title", "") or ""
    url = getattr(state, "url", "") or ""
    header = f"Page: {title[:120]} | {url[:200]}"
    return header + "\n- " + "\n- ".join(parts)
