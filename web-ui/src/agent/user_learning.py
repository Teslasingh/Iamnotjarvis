"""
Capture manual browser corrections (clicks, scrolls, navigation, form edits) as agent learnings.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from playwright.async_api import Page

_USER_ACTION_INIT_SCRIPT = """
() => {
  if (window.__buUserLearningInstalled) return;
  window.__buUserLearningInstalled = true;
  let lastScrollReport = 0;

  const isAgentEra = () => {
    const until = window.__buAgentActionUntil;
    return typeof until === 'number' && Date.now() < until;
  };

  const report = (payload) => {
    if (isAgentEra()) return;
    try {
      if (typeof window.__buUserAction === 'function') {
        window.__buUserAction(payload);
      }
    } catch (e) {}
  };

  const fieldLabel = (el) => {
    if (!el) return '';
    const id = el.id;
    if (id) {
      const lbl = document.querySelector(`label[for="${CSS.escape(id)}"]`);
      if (lbl) return (lbl.innerText || '').trim().slice(0, 80);
    }
    const labelled = el.closest('label');
    if (labelled) return (labelled.innerText || '').trim().slice(0, 80);
    return (
      el.getAttribute('aria-label') ||
      el.getAttribute('placeholder') ||
      el.getAttribute('name') ||
      ''
    ).trim().slice(0, 80);
  };

  const scrollPayload = (target) => {
    const isWin = !target || target === document || target === document.documentElement;
    const y = isWin
      ? Math.round(window.scrollY || document.documentElement.scrollTop || 0)
      : Math.round(target.scrollTop || 0);
    const container = isWin
      ? ''
      : (target.id ? '#' + target.id : (target.className ? '.' + String(target.className).split(/\\s+/)[0] : target.tagName));
    return { type: 'scroll', scrollY: y, container: String(container || '').slice(0, 80), url: location.href };
  };

  document.addEventListener('click', (e) => {
    const el = e.target;
    if (!el || !el.tagName) return;
    const link = el.closest('a');
    const text = (
      el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || ''
    ).trim().slice(0, 120);
    report({
      type: 'click',
      tag: el.tagName,
      text,
      href: (el.href || link?.href || '').slice(0, 200),
      role: el.getAttribute('role') || '',
      isTrusted: !!e.isTrusted,
      url: location.href,
    });
  }, true);

  const onScroll = (e) => {
    const now = Date.now();
    if (now - lastScrollReport < 700) return;
    lastScrollReport = now;
    report(scrollPayload(e.target));
  };
  window.addEventListener('scroll', onScroll, { passive: true, capture: true });
  document.addEventListener('scroll', onScroll, { passive: true, capture: true });

  const reportFieldEdit = (el, kind) => {
    if (!el || !el.tagName) return;
    const tag = el.tagName.toLowerCase();
    if (!['input', 'textarea', 'select'].includes(tag)) return;
    const inputType = (el.getAttribute('type') || '').toLowerCase();
    let value = '';
    if (inputType === 'password') {
      value = '[password]';
    } else if (tag === 'select') {
      value = (el.options[el.selectedIndex]?.text || el.value || '').trim().slice(0, 80);
    } else {
      value = (el.value || '').trim().slice(0, 80);
    }
    report({
      type: kind,
      field: fieldLabel(el),
      tag,
      value,
      url: location.href,
    });
  };

  document.addEventListener('input', (e) => reportFieldEdit(e.target, 'input'), true);
  document.addEventListener('change', (e) => reportFieldEdit(e.target, 'change'), true);
}
"""

_AGENT_NAV_ACTIONS = frozenset({
    "go_to_url",
    "open_tab",
    "switch_tab",
    "search_google",
    "go_back",
    "go_forward",
})

_AGENT_SCROLL_ACTIONS = frozenset({
    "scroll_down",
    "scroll_up",
    "scroll_to_bottom",
    "scroll_to_top",
    "scroll_to_text",
    "scroll_element_into_view",
    "scroll_to_edge",
    "reveal_more_content",
})

_AGENT_CLICK_ACTIONS = frozenset({
    "click_element_by_index",
    "click_element",
    "advance_form_step",
    "select_dropdown_option",
    "get_dropdown_options",
    "send_keys",
})

_SCROLL_DELTA_THRESHOLD = int(os.getenv("USER_LEARNING_SCROLL_THRESHOLD_PX", "80"))

_AGENT_SUPPRESSION_JS = "window.__buAgentActionUntil = Date.now() + {ms};"

_PAGE_SNAPSHOT_JS = r"""
() => {
  const cx = Math.floor(window.innerWidth / 2);
  const cy = Math.floor(window.innerHeight / 2);
  let hit = document.elementFromPoint(cx, cy);
  let containerScrollTop = 0;
  let containerSource = 'window';
  const check = (el, priority) => {
    if (!el || el.scrollHeight <= el.clientHeight + 8) return null;
    const r = el.getBoundingClientRect();
    if (!(r.left <= cx && r.right >= cx && r.top <= cy && r.bottom >= cy) && priority < 8) return null;
    const above = Math.round(el.scrollTop);
    const src = el.id ? '#' + el.id : (el.className ? '.' + String(el.className).split(/\s+/)[0] : el.tagName);
    return { scrollTop: above, source: String(src).slice(0, 80), priority };
  };
  let best = null;
  if (hit) {
    let cur = hit;
    while (cur && cur !== document.documentElement) {
      const role = cur.getAttribute && cur.getAttribute('role');
      const p = (role === 'dialog' || role === 'alertdialog') ? 10 : 5;
      const cand = check(cur, p);
      if (cand && (!best || cand.priority >= best.priority)) best = cand;
      cur = cur.parentElement;
    }
  }
  if (best) {
    containerScrollTop = best.scrollTop;
    containerSource = best.source;
  }
  return {
    url: location.href,
    title: document.title || '',
    scrollY: Math.round(window.scrollY || document.documentElement.scrollTop || 0),
    containerScrollTop,
    containerSource,
  };
}
"""


def user_learning_enabled() -> bool:
    return os.getenv("USER_LEARNING_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def agent_suppression_script(ms: int = 2000) -> str:
    return _AGENT_SUPPRESSION_JS.format(ms=int(ms))


def format_user_event(event: dict[str, Any]) -> Optional[str]:
    kind = (event.get("type") or "").lower()
    url = (event.get("url") or "")[:120]
    if kind == "click":
        text = (event.get("text") or "").strip()
        tag = (event.get("tag") or "").lower()
        href = (event.get("href") or "").strip()
        label = text or href or tag or "element"
        if href and text:
            return f'User clicked "{text}" ({tag}) → {href}'
        return f'User clicked "{label}" ({tag}) on {url or "page"}'
    if kind == "scroll":
        y = event.get("scrollY")
        if y is None:
            return None
        container = (event.get("container") or "").strip()
        if container:
            return f'User scrolled {container} to ~{y}px on {url or "page"}'
        return f"User scrolled to ~{y}px on {url or 'page'}"
    if kind in ("input", "change"):
        field = (event.get("field") or event.get("tag") or "field").strip()
        value = (event.get("value") or "").strip()
        if not value:
            return None
        verb = "edited" if kind == "input" else "set"
        return f'User {verb} field "{field}" → "{value}" on {url or "page"}'
    if kind == "assistant":
        return (event.get("message") or "").strip()[:400] or None
    if kind == "navigation":
        return (event.get("message") or "").strip()[:400] or None
    return None


def filter_agent_era_events(
    events: list[dict[str, Any]],
    last_agent_actions: set[str],
    step_start: Optional[float],
    step_end: Optional[float],
) -> list[dict[str, Any]]:
    """Drop events that occurred during the previous agent step when agent caused them."""
    if not events or step_start is None or step_end is None:
        return events

    had_click = bool(last_agent_actions & _AGENT_CLICK_ACTIONS)
    had_scroll = bool(last_agent_actions & _AGENT_SCROLL_ACTIONS)
    filtered: list[dict[str, Any]] = []

    for event in events:
        ts = event.get("ts")
        if ts is None:
            filtered.append(event)
            continue
        try:
            ts_f = float(ts)
        except (TypeError, ValueError):
            filtered.append(event)
            continue
        if not (step_start <= ts_f <= step_end):
            filtered.append(event)
            continue

        kind = (event.get("type") or "").lower()
        if kind == "click" and had_click:
            continue
        if kind == "scroll" and had_scroll:
            continue
        filtered.append(event)

    return filtered


def dedup_learning_lines(lines: list[str]) -> list[str]:
    """Collapse duplicate lines; prefer snapshot-delta scroll over redundant JS scroll."""
    if not lines:
        return []

    seen: set[str] = set()
    has_delta_scroll = any("scrolled" in ln and "manually" in ln for ln in lines)
    out: list[str] = []

    for line in lines:
        if line in seen:
            continue
        if has_delta_scroll and line.startswith("User scrolled") and "manually" not in line:
            continue
        seen.add(line)
        out.append(line)

    return out


async def read_page_snapshot(page: Page) -> dict[str, Any]:
    try:
        metrics = await page.evaluate(_PAGE_SNAPSHOT_JS)
        return metrics or {}
    except Exception:
        return {"url": getattr(page, "url", "") or ""}


def detect_snapshot_delta(
    previous: Optional[dict[str, Any]],
    current: dict[str, Any],
    last_agent_actions: set[str],
) -> list[str]:
    """Infer user corrections when the page changed without a matching agent action."""
    if not previous or not current:
        return []

    lines: list[str] = []
    prev_url = (previous.get("url") or "").strip()
    curr_url = (current.get("url") or "").strip()
    if prev_url and curr_url and prev_url != curr_url:
        if not (last_agent_actions & _AGENT_NAV_ACTIONS):
            lines.append(
                f"User navigated manually: {prev_url[:100]} → {curr_url[:100]}"
            )

    prev_y = int(previous.get("scrollY") or 0)
    curr_y = int(current.get("scrollY") or 0)
    window_scrolled = abs(curr_y - prev_y) >= _SCROLL_DELTA_THRESHOLD

    prev_c = int(previous.get("containerScrollTop") or 0)
    curr_c = int(current.get("containerScrollTop") or 0)
    container_scrolled = abs(curr_c - prev_c) >= _SCROLL_DELTA_THRESHOLD

    if (window_scrolled or container_scrolled) and not (last_agent_actions & _AGENT_SCROLL_ACTIONS):
        if window_scrolled:
            direction = "down" if curr_y > prev_y else "up"
            lines.append(
                f"User scrolled {direction} manually (~{prev_y}px → ~{curr_y}px)"
            )
        elif container_scrolled:
            direction = "down" if curr_c > prev_c else "up"
            src = (current.get("containerSource") or "modal").strip()
            lines.append(
                f"User scrolled {direction} inside {src} manually (~{prev_c}px → ~{curr_c}px)"
            )

    return lines


def user_action_init_script() -> str:
    return _USER_ACTION_INIT_SCRIPT
