"""
Generic UI defaults for web automation (not tied to a specific site or user).

Site-specific instructions, credentials, and profile data belong in prompt.txt / the task box.
"""

from __future__ import annotations

from typing import Optional

# Used when "Extend system prompt" is left empty (Agent Settings).
DEFAULT_EXTEND_SYSTEM_PROMPT = """
# Web automation (generic)

## Source of truth
- Follow the **user task prompt** for goals, URLs, credentials, filters, and field values.
- Do not invent or assume data that the task did not provide.

## Forms and multi-step flows
- Fill **only** required fields (required attribute, asterisk, or visible validation errors).
- Prefer dropdown tools when options are unclear: get_dropdown_options → select_dropdown_option.
- Do not re-enter fields that already show the correct value in the DOM or [Form progress].
- Advance with Next, Continue, Save, or Submit when the current step is complete.

## Page layout
- Work inside the active **modal, dialog, or drawer** when one is centered on screen.
- Prefer scroll_element_into_view on a target index; otherwise small scroll_down steps.
- If content above the viewport was skipped, scroll_up or scroll_to_top before scrolling down again.
- Dismiss blocking overlays (cookies, promos) when they prevent progress, unless the task requires them.

## Search and listings
- After search or opening a jobs page, allow results to load; use the full page briefing and OCR before clicking cards.
- Use ocr_search_visible_text or wait_for_search_results if listings or Apply links are not in the DOM list yet.

## Scope and recovery
- Stay within what the task asks; skip redundant, already-complete, or unnecessarily long flows.
- If progress stalls, change approach (scroll, another index, go_back, extract_content, ask_for_assistant).
"""

RUN_AGENT_TAB_BLURB = """
**Task prompt** loads from `web-ui/prompt.txt` — put your specific instructions there (saved on run).

**Goal-driven autonomy** (recommended): adapt to live UI state. **Remember session context** (recommended): carries applied/skipped jobs and past agent memory across steps and runs. **Multi-step form mode** (optional): richer form/OCR briefing.
"""


def prompt_textbox_lines(text: str, *, min_lines: int = 16, max_lines: int = 48) -> int:
    if not text:
        return min_lines
    return max(min_lines, min(max_lines, text.count("\n") + 2))


def resolve_extend_system_prompt(user_value: Optional[str]) -> Optional[str]:
    """Use UI extend text, or generic defaults when empty."""
    if (user_value or "").strip():
        return user_value.strip()
    stripped = DEFAULT_EXTEND_SYSTEM_PROMPT.strip()
    return stripped or None
