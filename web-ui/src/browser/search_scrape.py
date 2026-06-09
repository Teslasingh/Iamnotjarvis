"""Search / navigation: allow pages to settle, then run full scrape + OCR."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, FrozenSet, Optional

logger = logging.getLogger(__name__)

SEARCH_SETTLE_WAIT_S = float(os.getenv("SEARCH_SETTLE_WAIT_S", "2.5"))
SEARCH_FULL_SCRAPE = os.getenv("SEARCH_FULL_SCRAPE", "true").lower() in (
    "1", "true", "yes", "on",
)

_SEARCH_URL_RE = re.compile(
    r"(/search|/jobs?/|job-search|jobsearch|/recruit|/career|/vacanc|"
    r"keyword=|query=|q=|naukri\.com/search|linkedin\.com/jobs)",
    re.I,
)
_SEARCH_TASK_RE = re.compile(
    r"\b(search(?:ing)?|find jobs?|job listings?|filter|results?)\b",
    re.I,
)
_SEARCH_ACTION_NAMES = frozenset({
    "go_to_url",
    "click_element_by_index",
    "input_text",
    "send_keys",
    "open_tab",
    "switch_tab",
})


def search_full_scrape_enabled() -> bool:
    return SEARCH_FULL_SCRAPE


def url_looks_like_search_results(url: str) -> bool:
    return bool(url and _SEARCH_URL_RE.search(url))


def task_mentions_search(task: str) -> bool:
    return bool(task and _SEARCH_TASK_RE.search(task))


def actions_trigger_search_settle(action_names: FrozenSet[str]) -> bool:
    return bool(action_names & _SEARCH_ACTION_NAMES)


def should_run_full_search_scrape(
    *,
    url: str = "",
    pending: bool = False,
    task: str = "",
    step_number: int = 0,
) -> bool:
    if not search_full_scrape_enabled():
        return pending
    if pending:
        return True
    if url_looks_like_search_results(url):
        return True
    if step_number <= 2 and task_mentions_search(task):
        return True
    return False


async def settle_page_after_search(page, *, extra_wait_s: Optional[float] = None) -> None:
    """
    Give SPAs time to render search results before get_state / OCR / scrape.
    """
    wait_s = extra_wait_s if extra_wait_s is not None else SEARCH_SETTLE_WAIT_S
    if wait_s <= 0:
        return
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=int(wait_s * 1000) + 5000)
    except Exception as exc:
        logger.debug("search settle domcontentloaded: %s", exc)
    try:
        await page.wait_for_load_state("networkidle", timeout=int(wait_s * 1000) + 8000)
    except Exception as exc:
        logger.debug("search settle networkidle (continuing): %s", exc)
    await asyncio.sleep(wait_s)
    logger.info("Search page settle wait finished (%.1fs)", wait_s)


async def wait_for_results_hint(page, *, timeout_ms: int = 8000) -> Optional[str]:
    """Best-effort wait for job-card / result-list patterns."""
    selectors = [
        "[data-job-id]",
        ".jobTuple",
        ".job-card",
        ".jobs-search-results-list",
        ".jobs-search__results-list",
        "[class*='job-search']",
        "[class*='search-results']",
        "main [role='list']",
    ]
    for sel in selectors:
        try:
            await page.wait_for_selector(sel, state="visible", timeout=timeout_ms // len(selectors) + 500)
            return sel
        except Exception:
            continue
    return None
