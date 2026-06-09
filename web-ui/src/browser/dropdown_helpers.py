"""Robust dropdown selection: native <select>, nested selects, and custom comboboxes."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext

logger = logging.getLogger(__name__)

_NATIVE_OPTIONS_JS = r"""
(xpath) => {
  const select = document.evaluate(xpath, document, null,
    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
  if (!select || select.tagName.toLowerCase() !== 'select') return null;
  return {
    options: Array.from(select.options).map((opt, i) => ({
      text: opt.text,
      value: opt.value,
      index: i,
    })),
    id: select.id,
    name: select.name,
  };
}
"""

_NATIVE_SELECT_JS = r"""
(args) => {
  const xpath = args[0];
  const wanted = (args[1] || '').trim();
  const wantLower = wanted.toLowerCase();

  const select = document.evaluate(xpath, document, null,
    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
  if (!select || select.tagName.toLowerCase() !== 'select') {
    return { ok: false, error: 'not a select element' };
  }

  const score = (text) => {
    const t = (text || '').trim();
    const tl = t.toLowerCase();
    if (tl === wantLower) return 100;
    if (tl.includes(wantLower) || wantLower.includes(tl)) return 80;
    const words = wantLower.split(/\s+/).filter(Boolean);
    if (words.length && words.every((w) => tl.includes(w))) return 70;
    return 0;
  };

  let best = null;
  let bestScore = 0;
  for (const opt of select.options) {
    const s = score(opt.text);
    if (s > bestScore) {
      bestScore = s;
      best = opt;
    }
  }
  if (!best || bestScore < 70) {
    return {
      ok: false,
      available: Array.from(select.options).map((o) => o.text),
      wanted,
    };
  }

  select.value = best.value;
  select.selectedIndex = best.index;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return { ok: true, selected: best.text, value: best.value, score: bestScore };
}
"""

_FIND_NESTED_SELECT_JS = r"""
(xpath) => {
  const root = document.evaluate(xpath, document, null,
    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
  if (!root) return null;
  if (root.tagName && root.tagName.toLowerCase() === 'select') return { xpath: xpath, from: 'self' };
  const sel = root.querySelector && root.querySelector('select');
  if (!sel) return null;
  return { xpath: xpath, from: 'nested', hasSelect: true };
}
"""

_NATIVE_SELECT_FROM_TRIGGER_JS = r"""
(args) => {
  const xpath = args[0];
  const wanted = (args[1] || '').trim();
  const wantLower = wanted.toLowerCase();
  const root = document.evaluate(xpath, document, null,
    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
  if (!root) return { ok: false, error: 'root not found' };
  const select = root.tagName.toLowerCase() === 'select' ? root : root.querySelector('select');
  if (!select) return { ok: false, error: 'no select under trigger' };

  const score = (text) => {
    const tl = (text || '').trim().toLowerCase();
    if (tl === wantLower) return 100;
    if (tl.includes(wantLower) || wantLower.includes(tl)) return 80;
    return 0;
  };
  let best = null, bestScore = 0;
  for (const opt of select.options) {
    const s = score(opt.text);
    if (s > bestScore) { bestScore = s; best = opt; }
  }
  if (!best || bestScore < 70) {
    return { ok: false, available: Array.from(select.options).map((o) => o.text), wanted };
  }
  select.value = best.value;
  select.selectedIndex = best.index;
  select.dispatchEvent(new Event('input', { bubbles: true }));
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return { ok: true, selected: best.text, value: best.value, from: 'trigger' };
}
"""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _best_text_match(options: List[str], wanted: str, min_score: float = 0.55) -> Optional[str]:
    if not wanted or not options:
        return None
    best_text = None
    best_score = 0.0
    want_n = _normalize(wanted)
    for opt in options:
        opt_n = _normalize(opt)
        if opt_n == want_n:
            return opt
        if want_n in opt_n or opt_n in want_n:
            score = 0.85
        else:
            score = _similarity(opt, wanted)
        if score > best_score:
            best_score = score
            best_text = opt
    if best_score >= min_score:
        return best_text
    return None


async def _scroll_and_click(browser: BrowserContext, dom_element) -> Any:
    """Return Playwright element handle after scroll + click."""
    handle = await browser.get_locate_element(dom_element)
    if handle is None:
        return None
    try:
        await handle.scroll_into_view_if_needed(timeout=3000)
    except Exception:
        pass
    try:
        await handle.click(timeout=5000)
    except Exception:
        await handle.click(force=True, timeout=5000)
    return handle


async def _collect_custom_options(page, *, timeout_ms: int = 2500) -> List[str]:
    """Scrape visible options from open listbox/menu/combobox."""
    await asyncio.sleep(0.35)
    texts: List[str] = []
    try:
        found = await page.evaluate(
            r"""() => {
              const out = [];
              const seen = new Set();
              const add = (t) => {
                const s = (t || '').replace(/\s+/g, ' ').trim();
                if (s.length < 1 || s.length > 200 || seen.has(s)) return;
                seen.add(s);
                out.push(s);
              };
              const selectors = [
                '[role="listbox"] [role="option"]',
                '[role="listbox"] li',
                '[role="menu"] [role="menuitem"]',
                '[role="menu"] li',
                'ul[role="listbox"] li',
                '.rc-virtual-list-holder-inner div',
                '[class*="dropdown"] [class*="option"]',
                '[class*="select"] [class*="option"]',
                'li[aria-selected]',
                'div[aria-selected]',
              ];
              for (const sel of selectors) {
                document.querySelectorAll(sel).forEach((el) => {
                  const r = el.getBoundingClientRect();
                  if (r.width > 0 && r.height > 0) add(el.innerText || el.textContent);
                });
              }
              return out.slice(0, 80);
            }"""
        )
        if isinstance(found, list):
            texts.extend([str(t) for t in found if t])
    except Exception as exc:
        logger.debug("collect custom options failed: %s", exc)
    return texts


async def _click_option_by_text(page, text: str) -> bool:
    """Try Playwright locators to click a visible option."""
    want = text.strip()
    if not want:
        return False

    locators = [
        page.get_by_role("option", name=want, exact=False),
        page.get_by_role("menuitem", name=want, exact=False),
        page.get_by_text(want, exact=True),
        page.get_by_text(want, exact=False),
    ]

    for loc in locators:
        try:
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 8)):
            try:
                item = loc.nth(i)
                if await item.is_visible():
                    await item.scroll_into_view_if_needed(timeout=2000)
                    await item.click(timeout=3000)
                    return True
            except Exception:
                continue
    return False


async def select_custom_combobox(
    browser: BrowserContext,
    dom_element,
    text: str,
) -> Tuple[bool, str]:
    """Open combobox at index and pick option (LinkedIn, MUI, React Select, etc.)."""
    page = await browser.get_current_page()
    handle = await _scroll_and_click(browser, dom_element)
    if handle is None:
        return False, f"Could not click element at index {dom_element.highlight_index}"

    await asyncio.sleep(0.4)
    options = await _collect_custom_options(page)
    match = _best_text_match(options, text) if options else None
    target = match or text

    if await _click_option_by_text(page, target):
        return True, f"Selected custom dropdown option '{target}' (combobox)"

    # Type-ahead fallback
    try:
        await handle.click(timeout=3000)
        await asyncio.sleep(0.15)
        await handle.fill("")
        await page.keyboard.type(target, delay=30)
        await asyncio.sleep(0.35)
        if await _click_option_by_text(page, target):
            return True, f"Selected '{target}' via type-ahead + click"
        await page.keyboard.press("Enter")
        await asyncio.sleep(0.2)
        return True, f"Selected '{target}' via type-ahead + Enter"
    except Exception as exc:
        logger.debug("type-ahead dropdown failed: %s", exc)

    hint = f" visible options: {options[:15]}" if options else ""
    return False, f"Custom dropdown: could not select '{text}'.{hint}"


async def select_native_in_frames(
    page,
    xpath: str,
    text: str,
) -> Tuple[bool, str]:
    """Native <select> with fuzzy JS + Playwright fallbacks across frames."""
    for frame_index, frame in enumerate(page.frames):
        try:
            nested = await frame.evaluate(_FIND_NESTED_SELECT_JS, xpath)
            use_xpath = nested["xpath"] if nested else xpath

            if nested and nested.get("hasSelect"):
                trig = await frame.evaluate(_NATIVE_SELECT_FROM_TRIGGER_JS, [use_xpath, text])
                if trig and trig.get("ok"):
                    return True, (
                        f"Selected '{trig.get('selected')}' (nested select under trigger, frame {frame_index})"
                    )

            info = await frame.evaluate(_NATIVE_OPTIONS_JS, use_xpath)
            if info is None:
                continue

            result = await frame.evaluate(_NATIVE_SELECT_JS, [use_xpath, text])
            if result and result.get("ok"):
                return True, (
                    f"Selected '{result.get('selected')}' (native select, frame {frame_index})"
                )

            locator = frame.locator("//" + use_xpath).nth(0)
            try:
                await locator.scroll_into_view_if_needed(timeout=3000)
            except Exception:
                pass

            for label in (text, _best_text_match(
                [o["text"] for o in (info or {}).get("options", [])], text
            )):
                if not label:
                    continue
                try:
                    values = await locator.select_option(label=label, timeout=5000)
                    return True, f"Selected '{label}' via Playwright label (frame {frame_index}), values={values}"
                except Exception:
                    pass
                try:
                    values = await locator.select_option(value=label, timeout=3000)
                    return True, f"Selected value '{label}' (frame {frame_index})"
                except Exception:
                    pass

            available = (result or {}).get("available") or [
                o.get("text") for o in (info or {}).get("options", [])
            ]
            if available:
                pick = _best_text_match([str(a) for a in available], text)
                if pick:
                    try:
                        values = await locator.select_option(label=pick, timeout=5000)
                        return True, f"Selected fuzzy match '{pick}' (frame {frame_index})"
                    except Exception:
                        js2 = await frame.evaluate(_NATIVE_SELECT_JS, [use_xpath, pick])
                        if js2 and js2.get("ok"):
                            return True, f"Selected fuzzy '{js2.get('selected')}' via JS"

        except Exception as exc:
            logger.debug("native select frame %s failed: %s", frame_index, exc)

    return False, f"Could not select '{text}' in native <select> across frames"


async def get_dropdown_options_robust(
    browser: BrowserContext,
    index: int,
) -> ActionResult:
    page = await browser.get_current_page()
    selector_map = await browser.get_selector_map()
    dom_element = selector_map.get(index)
    if dom_element is None:
        return ActionResult(error=f"No element at index {index}")

    xpath = dom_element.xpath
    all_lines: List[str] = []

    for frame_index, frame in enumerate(page.frames):
        try:
            nested = await frame.evaluate(_FIND_NESTED_SELECT_JS, xpath)
            use_xpath = nested["xpath"] if nested else xpath
            info = await frame.evaluate(_NATIVE_OPTIONS_JS, use_xpath)
            if info and info.get("options"):
                for opt in info["options"]:
                    encoded = json.dumps(opt["text"])
                    all_lines.append(f'{opt["index"]}: text={encoded}')
                msg = "\n".join(all_lines)
                msg += "\nUse exact or close text in select_dropdown_option."
                return ActionResult(extracted_content=msg, include_in_memory=True)
        except Exception:
            continue

    # Custom combobox: open and list
    try:
        await _scroll_and_click(browser, dom_element)
        options = await _collect_custom_options(page)
        if options:
            for i, opt in enumerate(options):
                all_lines.append(f'{i}: text={json.dumps(opt)}')
            msg = "\n".join(all_lines)
            msg += "\nUse select_dropdown_option with matching text (custom combobox)."
            return ActionResult(extracted_content=msg, include_in_memory=True)
    except Exception as exc:
        logger.debug("get custom options failed: %s", exc)

    return ActionResult(
        extracted_content=(
            "No options found. Element may be a custom dropdown — "
            "use select_dropdown_option after scrolling it into view, or click then type text."
        ),
        include_in_memory=True,
    )


async def select_dropdown_option_robust(
    browser: BrowserContext,
    index: int,
    text: str,
) -> ActionResult:
    selector_map = await browser.get_selector_map()
    dom_element = selector_map.get(index)
    if dom_element is None:
        return ActionResult(error=f"No element at index {index}")

    page = await browser.get_current_page()
    xpath = dom_element.xpath
    tag = (dom_element.tag_name or "").lower()

    # 1) Native <select> (element or nested)
    ok, msg = await select_native_in_frames(page, xpath, text)
    if ok:
        logger.info(msg)
        return ActionResult(extracted_content=msg, include_in_memory=True)

    # 2) Custom combobox (div/button/input role=combobox)
    ok, msg = await select_custom_combobox(browser, dom_element, text)
    if ok:
        logger.info(msg)
        return ActionResult(extracted_content=msg, include_in_memory=True)

    # 3) Retry native on nested path only
    for frame in page.frames:
        try:
            nested = await frame.evaluate(_FIND_NESTED_SELECT_JS, xpath)
            if nested:
                ok, msg = await select_native_in_frames(page, nested["xpath"], text)
                if ok:
                    return ActionResult(extracted_content=msg, include_in_memory=True)
        except Exception:
            pass

    return ActionResult(
        error=f"Dropdown selection failed for '{text}' at index {index}. {msg}. "
        "Try get_dropdown_options first, scroll_element_into_view, then retry.",
        include_in_memory=True,
    )
