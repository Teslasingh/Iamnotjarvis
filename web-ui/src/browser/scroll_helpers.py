"""Scroll from viewport center so popups/modals at center are targeted, not the background page."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Smaller steps avoid skipping required fields at the top of long modals/forms.
_SCROLL_STEP_FRACTION = float(os.getenv("SCROLL_STEP_FRACTION", "0.35"))
_SCROLL_MAX_EDGE_STEPS = max(1, int(os.getenv("SCROLL_MAX_EDGE_STEPS", "3")))
_SCROLL_EDGE_STOP_BELOW_PX = int(os.getenv("SCROLL_EDGE_STOP_BELOW_PX", "48"))

_SCROLL_FROM_CENTER_JS = r"""
(args) => {
  const direction = args[0];
  const amount = args[1];
  const sign = direction === 'up' ? -1 : 1;
  const cx = Math.floor(window.innerWidth / 2);
  const cy = Math.floor(window.innerHeight / 2);

  const isScrollable = (el) => {
    if (!el || el === document.body || el === document.documentElement) return false;
    return el.scrollHeight > el.clientHeight + 8;
  };

  const pickBestScrollable = (root) => {
    if (!root) return null;
    let best = null;
    let bestScore = 0;
    const visit = (el, depth) => {
      if (!el || depth > 25) return;
      if (isScrollable(el)) {
        const r = el.getBoundingClientRect();
        const coversCenter = r.left <= cx && r.right >= cx && r.top <= cy && r.bottom >= cy;
        const role = el.getAttribute('role') || '';
        const modal = role === 'dialog' || role === 'alertdialog' || el.getAttribute('aria-modal') === 'true';
        const st = getComputedStyle(el);
        let score = r.width * r.height;
        if (coversCenter) score *= 3;
        if (modal) score *= 4;
        if (['auto', 'scroll', 'overlay'].includes(st.overflowY)) score *= 2;
        if (score > bestScore) { bestScore = score; best = el; }
      }
      for (const c of el.children || []) visit(c, depth + 1);
    };
    visit(root, 0);
    return best;
  };

  let hit = null;
  for (const [px, py] of [[cx, cy], [cx - 12, cy], [cx + 12, cy], [cx, cy - 12], [cx, cy + 12]]) {
    const el = document.elementFromPoint(px, py);
    if (el && el !== document.body && el !== document.documentElement) { hit = el; break; }
  }

  let centerContainer = null;
  if (hit) {
    try { if (hit.focus) hit.focus({ preventScroll: true }); } catch (e) {}
    const chain = [];
    let cur = hit;
    while (cur && cur !== document.documentElement) { chain.push(cur); cur = cur.parentElement; }
    for (const node of chain) {
      const role = node.getAttribute && node.getAttribute('role');
      const isModal = role === 'dialog' || role === 'alertdialog' ||
        node.getAttribute?.('aria-modal') === 'true' ||
        (node.className && String(node.className).toLowerCase().includes('modal'));
      if (isModal) {
        centerContainer = pickBestScrollable(node) || (isScrollable(node) ? node : null);
        if (centerContainer) break;
      }
    }
    if (!centerContainer) {
      for (const node of chain) {
        if (isScrollable(node)) {
          const r = node.getBoundingClientRect();
          if (r.left <= cx && r.right >= cx && r.top <= cy && r.bottom >= cy) {
            centerContainer = node; break;
          }
        }
      }
    }
    if (!centerContainer) centerContainer = pickBestScrollable(hit);
  }

  let delta = amount;
  const stepFrac = args[2] != null ? args[2] : 0.35;
  if (delta == null || delta === 0) delta = Math.round(window.innerHeight * stepFrac);
  delta = Math.abs(delta) * sign;

  const finish = (el, moved, method) => ({
    target: 'center-popup',
    tag: el.tagName,
    role: el.getAttribute('role') || '',
    moved,
    scrollTop: el.scrollTop,
    maxTop: Math.max(0, el.scrollHeight - el.clientHeight),
    method,
    centerX: cx,
    centerY: cy,
    hitTag: hit ? hit.tagName : null,
  });

  if (centerContainer) {
    const before = centerContainer.scrollTop;
    const maxTop = Math.max(0, centerContainer.scrollHeight - centerContainer.clientHeight);
    centerContainer.scrollTop = Math.max(0, Math.min(maxTop, before + delta));
    let moved = centerContainer.scrollTop - before;
    if (Math.abs(moved) >= 3) return finish(centerContainer, moved, 'center-scrollTop');

    centerContainer.scrollTop = before;
    try {
      centerContainer.dispatchEvent(new WheelEvent('wheel', {
        deltaY: delta, bubbles: true, cancelable: true, view: window, clientX: cx, clientY: cy,
      }));
    } catch (e) {}
    moved = centerContainer.scrollTop - before;
    if (Math.abs(moved) >= 3) return finish(centerContainer, moved, 'center-wheel');
    centerContainer.scrollTop = before;
  }

  const wBefore = window.scrollY;
  if (!centerContainer) {
    const se = document.scrollingElement || document.documentElement;
    window.scrollBy(0, delta);
    se.scrollTop = Math.max(0, se.scrollTop + delta);
    const wMoved = window.scrollY - wBefore;
    if (Math.abs(wMoved) >= 3) {
      return { target: 'window', moved: wMoved, scrollY: window.scrollY, method: 'window-fallback', centerX: cx, centerY: cy };
    }
  }

  return {
    target: 'none', moved: 0, method: 'none', centerX: cx, centerY: cy,
    hitTag: hit ? hit.tagName : null, containerTag: centerContainer ? centerContainer.tagName : null,
  };
}
"""

_SCROLL_EXTREME_JS = r"""
(edge) => {
  const cx = Math.floor(window.innerWidth / 2);
  const cy = Math.floor(window.innerHeight / 2);
  let hit = document.elementFromPoint(cx, cy);
  let centerContainer = null;
  const isScrollable = (el) => el && el.scrollHeight > el.clientHeight + 8;
  if (hit) {
    let cur = hit;
    while (cur && cur !== document.documentElement) {
      if (isScrollable(cur)) {
        const r = cur.getBoundingClientRect();
        if (r.left <= cx && r.right >= cx && r.top <= cy && r.bottom >= cy) {
          centerContainer = cur; break;
        }
      }
      const role = cur.getAttribute && cur.getAttribute('role');
      if (role === 'dialog' || role === 'alertdialog' || cur.getAttribute?.('aria-modal') === 'true') {
        if (isScrollable(cur)) { centerContainer = cur; break; }
      }
      cur = cur.parentElement;
    }
  }
  if (centerContainer) {
    const before = centerContainer.scrollTop;
    centerContainer.scrollTop = edge === 'bottom' ? centerContainer.scrollHeight : 0;
    const moved = centerContainer.scrollTop - before;
    if (Math.abs(moved) >= 3) {
      return { target: 'center-popup', tag: centerContainer.tagName, moved, scrollTop: centerContainer.scrollTop, method: 'center-edge', centerX: cx, centerY: cy };
    }
  }
  const se = document.scrollingElement || document.documentElement;
  const before = window.scrollY;
  window.scrollTo(0, edge === 'bottom' ? se.scrollHeight : 0);
  return { target: 'window', moved: window.scrollY - before, scrollY: window.scrollY, method: 'window-edge', centerX: cx, centerY: cy };
}
"""

_GET_REMAINING_JS = r"""
() => {
  const cx = Math.floor(window.innerWidth / 2);
  const cy = Math.floor(window.innerHeight / 2);
  let hit = document.elementFromPoint(cx, cy);
  let best = { below: 0, above: 0, source: 'window' };
  const check = (el, priority) => {
    if (!el || el.scrollHeight <= el.clientHeight + 8) return;
    const r = el.getBoundingClientRect();
    if (!(r.left <= cx && r.right >= cx && r.top <= cy && r.bottom >= cy) && priority < 8) return;
    const below = el.scrollHeight - el.clientHeight - el.scrollTop;
    const above = el.scrollTop;
    if (below > best.below || (priority >= 8 && below > 20)) {
      best = { below: Math.round(below), above: Math.round(above), source: el.tagName + (el.getAttribute('role') ? ':' + el.getAttribute('role') : '') };
    }
  };
  if (hit) {
    let cur = hit;
    while (cur && cur !== document.documentElement) {
      const role = cur.getAttribute && cur.getAttribute('role');
      const p = (role === 'dialog' || role === 'alertdialog') ? 10 : 5;
      check(cur, p);
      cur = cur.parentElement;
    }
  }
  const se = document.scrollingElement || document.documentElement;
  const wBelow = Math.max(0, se.scrollHeight - window.innerHeight - window.scrollY);
  if (wBelow > best.below && best.source === 'window') {
    best = { below: Math.round(wBelow), above: Math.round(window.scrollY), source: 'window' };
  }
  return best;
}
"""


async def get_viewport_center(page) -> Tuple[int, int]:
    """Viewport center in CSS pixels — where popups are usually anchored."""
    try:
        size = await page.evaluate(
            "() => ({ w: window.innerWidth, h: window.innerHeight })"
        )
        if size:
            return int(size["w"]) // 2, int(size["h"]) // 2
    except Exception:
        pass
    vp = page.viewport_size
    if vp:
        return vp["width"] // 2, vp["height"] // 2
    return 640, 550


async def _focus_center(page) -> None:
    """Focus element under viewport center so keyboard scroll hits the popup."""
    try:
        await page.evaluate(
            """() => {
              const cx = Math.floor(window.innerWidth / 2);
              const cy = Math.floor(window.innerHeight / 2);
              const el = document.elementFromPoint(cx, cy);
              if (el && el.focus) el.focus({ preventScroll: true });
            }"""
        )
    except Exception as exc:
        logger.debug("center focus failed: %s", exc)


async def _playwright_fallbacks(page, direction: str) -> Dict[str, Any]:
    """Move mouse to viewport center, then a small wheel nudge (no PageDown — too large)."""
    cx, cy = await get_viewport_center(page)
    method = []
    delta_y = 280 if direction == "down" else -280
    try:
        await page.mouse.move(cx, cy)
        method.append(f"mouse.move({cx},{cy})")
        await asyncio.sleep(0.05)
        await page.mouse.wheel(0, delta_y)
        method.append("mouse.wheel@center")
        await asyncio.sleep(0.2)
    except Exception as exc:
        logger.debug("center mouse.wheel failed: %s", exc)

    try:
        metrics = await page.evaluate(_GET_REMAINING_JS)
        return {
            "target": "fallback",
            "moved": "unknown",
            "method": "+".join(method),
            "remaining": metrics,
            "centerX": cx,
            "centerY": cy,
        }
    except Exception:
        return {"target": "fallback", "moved": 0, "method": "+".join(method), "centerX": cx, "centerY": cy}


async def scroll_by(page, direction: str, amount: Optional[int] = None) -> Dict[str, Any]:
    """Scroll the popup/modal under viewport center; fallback to window only if none."""
    result = await page.evaluate(
        _SCROLL_FROM_CENTER_JS, [direction, amount, _SCROLL_STEP_FRACTION]
    )
    result = result or {}
    moved = result.get("moved", 0)
    if isinstance(moved, (int, float)) and abs(moved) < 5:
        fb = await _playwright_fallbacks(page, direction)
        result["fallback"] = fb
        if result.get("target") == "none":
            result["target"] = "fallback"
            result["method"] = (result.get("method") or "") + "+" + fb.get("method", "")
    return result


async def scroll_to_edge(page, edge: str) -> Dict[str, Any]:
    """Top: jump to start. Bottom: incremental steps so top-of-form content is not skipped."""
    if edge not in ("top", "bottom"):
        edge = "bottom"
    if edge == "top":
        result = await page.evaluate(_SCROLL_EXTREME_JS, edge)
        result = result or {}
        if abs(result.get("moved", 0) or 0) < 5:
            await _playwright_fallbacks(page, "up")
            result["fallback_applied"] = True
        return result

    last: Dict[str, Any] = {}
    for step in range(_SCROLL_MAX_EDGE_STEPS):
        last = await scroll_by(page, "down", None)
        remaining = await get_remaining_scroll(page)
        below = int(remaining.get("below") or 0)
        last["edge_steps"] = step + 1
        if below < _SCROLL_EDGE_STOP_BELOW_PX:
            break
    return last


async def scroll_aggressive(page, direction: str = "down", max_attempts: int = 1) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    for i in range(max_attempts):
        last = await scroll_by(page, direction)
        moved = last.get("moved", 0)
        if isinstance(moved, (int, float)) and abs(moved) >= 5:
            last["attempts_used"] = i + 1
            return last
        await asyncio.sleep(0.25)
    last["attempts_used"] = max_attempts
    return last


async def get_remaining_scroll(page) -> Dict[str, Any]:
    try:
        return await page.evaluate(_GET_REMAINING_JS) or {}
    except Exception:
        return {"below": 0, "above": 0}


async def auto_reveal_hidden_content(page, *, threshold: int = 120) -> Optional[str]:
    """Single gentle scroll — only when a lot of content is below and little above."""
    metrics = await get_remaining_scroll(page)
    below = int(metrics.get("below") or 0)
    above = int(metrics.get("above") or 0)
    if below < threshold or above > 80:
        return None

    result = await scroll_by(page, "down", None)
    msg = format_scroll_result(result, "Auto-scrolled down one step from viewport center")
    remaining = await get_remaining_scroll(page)
    rb = int(remaining.get("below") or 0)
    ra = int(remaining.get("above") or 0)
    msg += f" | ~{rb}px below, ~{ra}px above ({remaining.get('source', '?')})"
    if ra > 100:
        msg += " — use scroll_to_top or scroll_up if you missed fields at the top."
    return msg


def format_scroll_result(result: Dict[str, Any], label: str) -> str:
    target = result.get("target", "unknown")
    moved = result.get("moved", 0)
    method = result.get("method", "")
    center = ""
    if result.get("centerX") is not None:
        center = f" @ center ({result.get('centerX')},{result.get('centerY')})"
    extra = f" via {method}" if method else ""

    if target == "center-popup":
        role = result.get("role") or result.get("tag") or "popup"
        return (
            f"{label} in centered {role}{center}{extra} (moved {moved}px, "
            f"scrollTop={result.get('scrollTop', '?')}/{result.get('maxTop', '?')})"
        )
    if target == "fallback":
        fb = result.get("fallback") or result
        return f"{label}{center} using {fb.get('method', 'wheel@center')} (remaining: {fb.get('remaining', {})})"
    if target == "none":
        return f"{label}{center}: no movement — popup at center may need reveal_more_content"
    return f"{label} on background page{center}{extra} (moved {moved}px)"
