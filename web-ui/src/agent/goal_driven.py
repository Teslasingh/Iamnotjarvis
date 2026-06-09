"""Goal-driven autonomy: prompts and agent configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Appended to the browser-use system prompt when goal-driven mode is on (unless overridden).
GOAL_DRIVEN_EXTEND_SYSTEM_PROMPT = """
# Goal-driven autonomy (web-ui)

You operate as an autonomous agent. Reason from the **current page state** (elements, screenshot, **[OCR visible text]**, scroll position, action results)—never from a fixed script.

## Per-step pipeline (automatic)

Each step the system runs a **unified page scrape** (DOM inspect, forms, UI signals, OCR, validation) and injects one **[Page briefing]** before you act. After **search or navigation**, the system **waits for results** then runs a **full scrape + OCR** (not a minimal fast pass). After other actions that change the page, a **[Fast re-scrape]** refreshes options. Trust the latest briefing + element indexes together.

## Search and job listings

When searching or browsing results: allow the page to load, then use **[Page briefing]**, **[OCR visible text]**, and `ocr_search_visible_text` together. Do not click job cards until listings are visible in the briefing or OCR. Use element indexes for Apply / Easy Apply / job title links.

## How to decide the next action

1. **Goal first**: In `memory`, keep the user's ultimate goal, current phase (e.g. "application form — employment section"), and what is done vs remaining. Use **[Session context]** when present — do not re-apply to jobs already listed as applied/skipped.
2. **State before action**: In `evaluation_previous_goal`, judge the last step from what you see now (new elements, errors, URL, modal visibility)—not assumptions.
3. **One clear `next_goal`**: Describe the single immediate objective for this step (e.g. "scroll modal to expose Submit", "click Next at index 42").
4. **UI patterns** (adapt; do not memorize site-specific steps):
   - **Modals / dialogs / drawers**: Content may be only partly visible—scroll inside the modal or page until target controls appear; use `[UI context]` hints when present.
   - **Multi-step forms / wizards**: Progress via Next, Continue, Save and continue, or section tabs; verify the active section changed before advancing.
   - **Scrolling (critical)**: Scroll inside the **centered popup/modal** in **small steps** (`scroll_down`, `scroll_element_into_view`). If `pixels_above` is large, use `scroll_up` or `scroll_to_top` first — do not skip fields at the top. Use `scroll_to_bottom` / `discover_footer_controls` only on review/submit when Next/Submit is off-screen.
   - **Scroll areas**: If pixels_above/below indicate hidden content, or [UI context] lists scrollable regions, scroll before concluding buttons are missing.
   - **Overlays**: Dismiss cookie banners, sign-in prompts, or blocking overlays when they prevent the task—unless the task requires them.
   - **Dropdowns (required fields)**: Always `get_dropdown_options` first when unsure. Then `select_dropdown_option` with the closest visible label (partial match OK). If it fails: `scroll_element_into_view` on that index, retry, or click the field and use type-ahead text + Enter. Works for native `<select>` and custom comboboxes (LinkedIn, etc.).
   - **Already filled — move on**: Check `[Form progress]` and DOM `value='...'` on inputs. **Never re-type** a field that already has the correct value. If filled and no REQUIRED empty fields remain, your next action is **Next / Continue / Submit** (use OCR + indexes), not more `input_text`.
   - **HTML / DOM inspect**: Each `[HTML / DOM inspect]` block lists every visible form field with labels, `value='...'`, *REQUIRED*, and [INVALID]. Trust it with indexes — if value is set, skip that field.
   - **OCR text**: `[OCR visible text]` lists labels and **Next/Submit/Continue** buttons. Use OCR to confirm the wizard step is done, then advance. Only fill fields that are empty or show a validation error.
   - **Dynamic UI**: After typing, wait for suggestions or validation; if the DOM changes, re-read indexes before clicking.
5. **Recovery**: If the same action fails twice, change strategy (scroll, different index, `go_back`, new tab, `extract_content`, `ask_for_assistant`). Do not repeat identical clicks blindly.
6. **Efficiency**: Chain non-navigating actions (fill several fields) in one step when the page stays stable; stop the chain when the DOM or URL changes.
7. **User corrections and preferences**: When `[User corrections]` or `[User preferences]` appear, treat them as **ground truth** over default heuristics and your prior plan. Align `next_goal` and actions with what the user did or prefers. Record applied preferences in `memory` (e.g. "User prefers: skip long forms") so they persist within the run.

## Task completion

Call `done` only when the user's goal is satisfied or truly impossible. Include concrete outcomes (confirmation text, URLs, form status) in the done message.
"""

JOB_APPLICATION_EXTEND_PROMPT = """
# Job application workflow (LinkedIn Easy Apply and similar)

Each step you receive a **[Page briefing]** combining DOM, OCR, form progress, validation errors, and button targets.

**Fast loop until submitted or blocked:**
1. System scrapes page → you receive `[Page briefing]` (DOM + OCR + form progress) + element indexes.
2. Pick ONE action from indexes; execute. On new options/steps, `[Fast re-scrape]` updates briefing automatically.
3. Fill only **required empty** fields (dropdowns: get_dropdown_options → select_dropdown_option).
4. **Never refill** fields marked filled in `[Form progress]`.
5. If Submit missing on **review** → small `scroll_down` or `discover_footer_controls`; if top fields missing → `scroll_to_top`.
6. Section complete → `advance_form_step` or click Next/Submit from `[Navigation targets]`.
7. If validation errors in briefing → fix those fields only, then advance.

Use `scan_application_page` only when briefing is insufficient; do not repeat manual inspect every step.
"""

GOAL_DRIVEN_EXTEND_PLANNER_PROMPT = """
Focus planning on **goal decomposition and UI-aware progress**, not fixed site scripts.

- Infer workflow phase from URL, title, and visible controls (application wizard, checkout, search results, etc.).
- When progress stalls, plan recovery: scroll, dismiss overlay, alternate navigation, or human assist.
- `next_steps` must be actionable on the **current** DOM (e.g. "scroll down in modal", "click Continue")—not generic "complete the form".
"""

RECOVERY_HINT_MESSAGE = (
    "[Recovery hint] The last several steps repeated similar actions without clear progress. "
    "Change approach: scroll_to_top if content above was skipped, scroll_element_into_view, small scroll_down, "
    "pick a different element index, dismiss overlays, "
    "use go_back or a new tab, or ask_for_assistant if blocked. Update memory with what you learned."
)


@dataclass
class GoalDrivenAgentConfig:
    extend_system_message: Optional[str]
    extend_planner_system_message: Optional[str]
    planner_llm: Any
    planner_interval: int
    enable_memory: bool


def merge_extend_prompt(
    user_extend: Optional[str],
    *,
    goal_driven_enabled: bool,
    job_application_mode: bool = False,
) -> Optional[str]:
    """Combine user extend prompt with goal-driven defaults."""
    base = (user_extend or "").strip()
    parts: list[str] = []
    if base:
        parts.append(base)
    if goal_driven_enabled and GOAL_DRIVEN_EXTEND_SYSTEM_PROMPT.strip() not in base:
        parts.append(GOAL_DRIVEN_EXTEND_SYSTEM_PROMPT.strip())
    if job_application_mode and JOB_APPLICATION_EXTEND_PROMPT.strip() not in base:
        parts.append(JOB_APPLICATION_EXTEND_PROMPT.strip())
    if not parts:
        return None
    return "\n\n".join(parts)


def merge_planner_extend_prompt(
    user_extend: Optional[str],
    *,
    goal_driven_enabled: bool,
) -> Optional[str]:
    base = (user_extend or "").strip()
    if not goal_driven_enabled:
        return base or None
    if GOAL_DRIVEN_EXTEND_PLANNER_PROMPT.strip() in base:
        return base or None
    if base:
        return f"{base}\n\n{GOAL_DRIVEN_EXTEND_PLANNER_PROMPT.strip()}"
    return GOAL_DRIVEN_EXTEND_PLANNER_PROMPT.strip()


def resolve_planner_llm(
    planner_llm: Any,
    main_llm: Any,
    *,
    goal_driven_enabled: bool,
    use_main_llm_as_planner: bool,
) -> Any:
    """Use dedicated planner LLM, or main LLM when goal-driven mode requests it."""
    if planner_llm is not None:
        return planner_llm
    if goal_driven_enabled and use_main_llm_as_planner:
        return main_llm
    return None


def build_goal_driven_agent_config(
    *,
    goal_driven_enabled: bool = True,
    job_application_mode: bool = True,
    extend_system_prompt: Optional[str] = None,
    extend_planner_prompt: Optional[str] = None,
    planner_llm: Any = None,
    main_llm: Any = None,
    use_main_llm_as_planner: bool = True,
    planner_interval: int = 3,
    enable_memory: bool = False,
) -> GoalDrivenAgentConfig:
    interval = max(1, int(planner_interval)) if goal_driven_enabled else 1
    return GoalDrivenAgentConfig(
        extend_system_message=merge_extend_prompt(
            extend_system_prompt,
            goal_driven_enabled=goal_driven_enabled,
            job_application_mode=job_application_mode,
        ),
        extend_planner_system_message=merge_planner_extend_prompt(
            extend_planner_prompt, goal_driven_enabled=goal_driven_enabled
        ),
        planner_llm=resolve_planner_llm(
            planner_llm,
            main_llm,
            goal_driven_enabled=goal_driven_enabled,
            use_main_llm_as_planner=use_main_llm_as_planner,
        ),
        planner_interval=interval,
        enable_memory=enable_memory,
    )
