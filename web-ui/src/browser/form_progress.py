"""Detect filled vs empty fields and advise when to advance (Next/Submit)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from browser_use.browser.views import BrowserState

_ADVANCE_RE = re.compile(
    r"\b(next|continue|submit|apply|save\s+and\s+continue|review|done|finish|proceed)\b",
    re.IGNORECASE,
)
_VALUE_RE = re.compile(r"value='([^']*)'|value=\"([^\"]*)\"", re.IGNORECASE)
_TYPE_RE = re.compile(r"type='([^']*)'|type=\"([^\"]*)\"", re.IGNORECASE)
_EMPTY_VALUES = frozenset({"", "select", "choose", "please select", "select an option", "—", "-"})

_FIELD_PROBE_JS = r"""
() => {
  const out = { filled: [], empty: [], required_empty: [] };
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 2 && r.height > 2 && r.bottom > 0 && r.top < window.innerHeight;
  };
  const labelOf = (el) => {
    const parts = [];
    if (el.labels && el.labels.length) parts.push(el.labels[0].innerText);
    parts.push(
      el.getAttribute('aria-label'),
      el.getAttribute('placeholder'),
      el.getAttribute('name'),
      el.id,
    );
    return (parts.find((p) => p && String(p).trim()) || el.tagName).trim().slice(0, 80);
  };
  const isEmpty = (el) => {
    const v = (el.value ?? '').trim();
    if (el.tagName === 'SELECT') {
      const opt = el.options[el.selectedIndex];
      const t = (opt && (opt.text || opt.label)) || v;
      if (!t || /select|choose|please/i.test(t)) return true;
      return false;
    }
    if (el.tagName === 'TEXTAREA') return v.length < 1;
    if (el.type === 'checkbox' || el.type === 'radio') return !el.checked;
    if (el.type === 'file') return !el.files || el.files.length === 0;
    return v.length < 1;
  };
  for (const el of document.querySelectorAll('input, select, textarea')) {
    if (!visible(el)) continue;
    const lab = labelOf(el);
    const req = el.required || el.getAttribute('aria-required') === 'true' ||
      (el.closest('[class*="required" i], [aria-required="true"]') != null);
    const entry = { label: lab, type: el.type || el.tagName, value: (el.value || '').slice(0, 60) };
    if (isEmpty(el)) {
      out.empty.push(entry);
      if (req) out.required_empty.push(entry);
    } else {
      out.filled.push(entry);
    }
  }
  return out;
}
"""


def _parse_fields_from_element_tree(elements_text: str) -> Dict[str, List[str]]:
    """Heuristic parse of browser-use element lines for input values."""
    filled: List[str] = []
    empty: List[str] = []
    for line in elements_text.split("\n"):
        lower = line.lower()
        if not any(t in lower for t in ("<input", "<textarea", "<select", "combobox", "textbox")):
            continue
        vm = _VALUE_RE.search(line)
        value = (vm.group(1) or vm.group(2) or "").strip() if vm else ""
        label = line[:120]
        if value and value.lower() not in _EMPTY_VALUES and len(value) > 0:
            filled.append(f"{label[:80]} → '{value[:40]}'")
        elif "<input" in lower or "<select" in lower or "combobox" in lower:
            empty.append(label[:80])
    return {"filled": filled[:12], "empty": empty[:12]}


def _ocr_advance_buttons(ocr_lines: List[str]) -> List[str]:
    found: List[str] = []
    for text in ocr_lines:
        if _ADVANCE_RE.search(text) and text not in found:
            found.append(text.strip())
    return found[:8]


async def probe_form_fields(page) -> Dict[str, Any]:
    try:
        return await page.evaluate(_FIELD_PROBE_JS) or {}
    except Exception:
        return {}


def build_form_progress_message(
    state: "BrowserState",
    *,
    dom_probe: Optional[Dict[str, Any]] = None,
    ocr_line_texts: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Tell the agent which fields look filled vs empty and when to click Next/Submit.
    """
    parts: List[str] = []

    try:
        elements_text = state.element_tree.clickable_elements_to_string()
    except Exception:
        elements_text = ""

    tree_parse = _parse_fields_from_element_tree(elements_text) if elements_text else {}

    filled_dom = list(dom_probe.get("filled") or []) if dom_probe else []
    empty_dom = list(dom_probe.get("empty") or []) if dom_probe else []
    req_empty = list(dom_probe.get("required_empty") or []) if dom_probe else []

    if filled_dom:
        for f in filled_dom[:8]:
            parts.append(f"  ✓ filled: {f.get('label', '?')} ({f.get('type', '')})")
    elif tree_parse.get("filled"):
        for x in tree_parse["filled"][:6]:
            parts.append(f"  ✓ filled (DOM): {x}")

    if req_empty:
        for f in req_empty[:6]:
            parts.append(f"  ✗ REQUIRED empty: {f.get('label', '?')}")
    elif empty_dom:
        for f in empty_dom[:5]:
            parts.append(f"  ○ empty: {f.get('label', '?')}")
    elif tree_parse.get("empty"):
        for x in tree_parse["empty"][:5]:
            parts.append(f"  ○ empty (DOM): {x}")

    advance = _ocr_advance_buttons(ocr_line_texts or [])
    for line in elements_text.split("\n"):
        if _ADVANCE_RE.search(line):
            m = re.search(r"\[(\d+)\]", line)
            if m:
                advance.append(f"[{m.group(1)}] {line.strip()[:60]}")

    if not parts and not advance:
        return None

    msg = "[Form progress — do NOT re-type fields that are already filled]\n"

    if filled_dom or tree_parse.get("filled"):
        msg += "Already filled — skip input_text on these; only fix if validation error appears.\n"

    if req_empty:
        msg += "Fill REQUIRED empty fields first, then advance.\n"
    elif not empty_dom and not tree_parse.get("empty") and (filled_dom or tree_parse.get("filled")):
        msg += "No empty inputs detected on screen — prefer Next/Continue/Submit if OCR/DOM shows them.\n"
    elif not req_empty and filled_dom and len(empty_dom) <= 1:
        msg += "Most fields filled — scroll if needed, then click Next/Continue/Submit.\n"

    if parts:
        msg += "\n".join(parts) + "\n"

    if advance:
        msg += "\n[Advance controls — click when current section is complete]\n- " + "\n- ".join(advance)

    msg += (
        "\n[OCR workflow] Use OCR button labels above with element indexes. "
        "If a field shows a value in DOM/OCR, move on — do not refill."
    )
    return msg
