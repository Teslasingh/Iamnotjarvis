"""
Deep HTML/DOM inspection — structured page parse beyond clickable element indexes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_CHARS = int(os.getenv("HTML_INSPECT_MAX_CHARS", "4500"))
_DEEP_MAX_CHARS = int(os.getenv("HTML_INSPECT_DEEP_MAX_CHARS", "12000"))

_DOM_INSPECT_JS = r"""
() => {
  const vis = (el) => {
    if (!el || el.nodeType !== 1) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || st.opacity === '0') return false;
    return r.bottom > 0 && r.top < window.innerHeight && r.right > 0 && r.left < window.innerWidth;
  };

  const labelFor = (el) => {
    const parts = [];
    if (el.labels && el.labels.length) {
      for (const lb of el.labels) parts.push((lb.innerText || '').trim());
    }
    parts.push(
      el.getAttribute('aria-label'),
      el.getAttribute('placeholder'),
      el.getAttribute('title'),
      el.getAttribute('name'),
      el.id,
    );
    const aria = el.getAttribute('aria-labelledby');
    if (aria) {
      const ref = document.getElementById(aria);
      if (ref) parts.push((ref.innerText || '').trim());
    }
    return parts.filter(Boolean).join(' | ').slice(0, 120);
  };

  const forms = [];
  document.querySelectorAll('form').forEach((form, fi) => {
    if (!vis(form) && !form.querySelector('input, select, textarea')) return;
    const fields = [];
    form.querySelectorAll('input, select, textarea, [role="combobox"], [role="listbox"]').forEach((el) => {
      if (!vis(el) && el.type !== 'hidden') return;
      const tag = el.tagName.toLowerCase();
      let value = '';
      if (tag === 'select') {
        const opt = el.options[el.selectedIndex];
        value = opt ? (opt.text || opt.value) : '';
      } else if (el.type === 'checkbox' || el.type === 'radio') {
        value = el.checked ? 'checked' : 'unchecked';
      } else {
        value = (el.value || '').slice(0, 80);
      }
      fields.push({
        tag,
        type: el.type || el.getAttribute('role') || '',
        name: el.name || '',
        id: el.id || '',
        label: labelFor(el),
        value,
        required: !!(el.required || el.getAttribute('aria-required') === 'true'),
        disabled: !!el.disabled,
        invalid: el.getAttribute('aria-invalid') === 'true',
      });
    });
    if (fields.length) forms.push({ index: fi, id: form.id || '', name: form.name || '', fields: fields.slice(0, 40) });
  });

  const orphanFields = [];
  document.querySelectorAll('input, select, textarea, [role="combobox"]').forEach((el) => {
    if (el.closest('form')) return;
    if (!vis(el)) return;
    orphanFields.push({
      tag: el.tagName.toLowerCase(),
      type: el.type || el.getAttribute('role') || '',
      label: labelFor(el),
      value: (el.value || '').slice(0, 60),
      required: !!(el.required || el.getAttribute('aria-required') === 'true'),
    });
  });

  const dialogs = [];
  document.querySelectorAll('[role="dialog"], [role="alertdialog"], dialog[open], [aria-modal="true"]').forEach((d) => {
    const r = d.getBoundingClientRect();
    if (r.width < 50) return;
    dialogs.push({
      role: d.getAttribute('role') || d.tagName,
      id: d.id || '',
      label: (d.getAttribute('aria-label') || '').slice(0, 80),
      text: (d.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 200),
    });
  });

  const headings = [];
  document.querySelectorAll('h1,h2,h3,h4').forEach((h) => {
    if (!vis(h)) return;
    headings.push({ level: h.tagName, text: (h.innerText || '').trim().slice(0, 100) });
  });

  const landmarks = [];
  for (const sel of ['main', '[role="main"]', 'nav', '[role="navigation"]', '[role="banner"]']) {
    document.querySelectorAll(sel).forEach((el) => {
      if (!vis(el)) return;
      landmarks.push({ tag: el.tagName, role: el.getAttribute('role') || '', text: (el.innerText || '').slice(0, 150).replace(/\s+/g, ' ') });
    });
  }

  const links = [];
  document.querySelectorAll('a[href]').forEach((a) => {
    if (!vis(a)) return;
    const t = (a.innerText || a.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim();
    if (t.length < 2) return;
    links.push({ text: t.slice(0, 60), href: (a.getAttribute('href') || '').slice(0, 80) });
  });

  const buttons = [];
  document.querySelectorAll('button, [role="button"], input[type="submit"], input[type="button"]').forEach((b) => {
    if (!vis(b)) return;
    buttons.push({
      text: (b.innerText || b.value || b.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim().slice(0, 60),
      type: b.type || b.getAttribute('role') || b.tagName,
      disabled: !!b.disabled,
    });
  });

  const meta = {
    url: location.href,
    title: document.title,
    readyState: document.readyState,
    iframeCount: document.querySelectorAll('iframe').length,
    formCount: forms.length,
    dialogCount: dialogs.length,
    visibleFieldCount: forms.reduce((n, f) => n + f.fields.length, 0) + orphanFields.length,
  };

  return {
    meta,
    forms: forms.slice(0, 8),
    orphanFields: orphanFields.slice(0, 25),
    dialogs: dialogs.slice(0, 6),
    headings: headings.slice(0, 15),
    landmarks: landmarks.slice(0, 8),
    links: links.slice(0, 20),
    buttons: buttons.slice(0, 25),
  };
}
"""


async def probe_dom_structure(page) -> Dict[str, Any]:
    try:
        data = await page.evaluate(_DOM_INSPECT_JS)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("DOM inspect failed: %s", exc)
        return {}


async def extract_readable_html_text(page) -> Optional[str]:
    """Main-content text extraction from full HTML (trafilatura-style)."""
    try:
        html = await page.content()
        if not html or len(html) < 50:
            return None
        from main_content_extractor import MainContentExtractor

        extractor = MainContentExtractor()
        text = extractor.extract(html)
        if text and isinstance(text, str):
            return text.strip()[:2500]
    except Exception as exc:
        logger.debug("MainContentExtractor failed: %s", exc)
    return None


def _format_field_line(f: Dict[str, Any]) -> str:
    req = " *REQUIRED*" if f.get("required") else ""
    dis = " [disabled]" if f.get("disabled") else ""
    inv = " [INVALID]" if f.get("invalid") else ""
    val = f.get("value", "")
    val_part = f" value='{val}'" if val else " (empty)"
    label = f.get("label") or f.get("name") or f.get("id") or f.get("type") or "field"
    return f"  - {label}{req}{val_part}{dis}{inv}"


def format_dom_inspection(data: Dict[str, Any], *, max_chars: int = _MAX_CHARS) -> str:
    if not data:
        return "[HTML inspect] No DOM structure returned."

    lines: List[str] = [
        "[HTML / DOM inspect — full page structure; use with element indexes for actions]",
    ]
    meta = data.get("meta") or {}
    if meta:
        lines.append(
            f"Document: {meta.get('title', '')[:100]} | forms={meta.get('formCount', 0)} "
            f"dialogs={meta.get('dialogCount', 0)} visibleFields={meta.get('visibleFieldCount', 0)} "
            f"iframes={meta.get('iframeCount', 0)}"
        )

    for d in data.get("dialogs") or []:
        lines.append(
            f"Dialog: role={d.get('role')} id={d.get('id')} label={d.get('label')} "
            f"snippet={d.get('text', '')[:80]}"
        )

    for form in data.get("forms") or []:
        lines.append(f"Form #{form.get('index', 0)} id={form.get('id')} name={form.get('name')}")
        for field in form.get("fields") or []:
            lines.append(_format_field_line(field))

    orphans = data.get("orphanFields") or []
    if orphans:
        lines.append("Fields outside <form> (common in SPAs / LinkedIn modals):")
        for field in orphans:
            lines.append(_format_field_line(field))

    headings = data.get("headings") or []
    if headings:
        lines.append("Headings: " + " | ".join(
            f"{h.get('level')}: {h.get('text', '')[:50]}" for h in headings[:8]
        ))

    buttons = data.get("buttons") or []
    nav_btns = [b for b in buttons if re.search(
        r"next|submit|continue|apply|review|save|cancel|close", b.get("text", ""), re.I
    )]
    if nav_btns:
        lines.append("Buttons (HTML visible):")
        for b in nav_btns[:12]:
            dis = " [disabled]" if b.get("disabled") else ""
            lines.append(f"  - \"{b.get('text')}\" ({b.get('type')}){dis}")
    elif buttons:
        lines.append("Buttons (sample): " + ", ".join(
            f"\"{b.get('text', '')[:30]}\"" for b in buttons[:8]
        ))

    links = data.get("links") or []
    if links:
        lines.append("Links (sample): " + ", ".join(
            f"{l.get('text', '')[:25]}" for l in links[:6]
        ))

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(HTML inspect truncated)"
    return text


async def build_html_inspection_briefing(
    page,
    *,
    include_readable_text: bool = True,
    deep: bool = False,
) -> str:
    data = await probe_dom_structure(page)
    max_c = _DEEP_MAX_CHARS if deep else _MAX_CHARS
    out = format_dom_inspection(data, max_chars=max_c)

    if include_readable_text:
        readable = await extract_readable_html_text(page)
        if readable:
            out += "\n\n[Readable page text from HTML extract]\n" + readable[:2000]

    if deep:
        try:
            snippet = await page.evaluate(
                """() => {
                  const modal = document.querySelector('[role="dialog"], [aria-modal="true"]');
                  const root = modal || document.body;
                  return root.innerHTML.length > 50000
                    ? root.innerHTML.slice(0, 8000) + '...'
                    : root.innerHTML;
                }"""
            )
            if snippet and isinstance(snippet, str):
                clean = re.sub(r"\s+", " ", snippet)[:3000]
                out += "\n\n[HTML snippet — modal or body, compressed]\n" + clean
        except Exception:
            pass

    return out


async def run_deep_html_inspect(page) -> str:
    return await build_html_inspection_briefing(page, include_readable_text=True, deep=True)
