import asyncio
import base64
import logging
import os
import time
from typing import Awaitable, Optional, Tuple, TypeVar

from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.browser.context import BrowserContextState
from browser_use.browser.views import BrowserError, BrowserState
from browser_use.dom.service import DomService
from playwright.async_api import Page

from src.agent.user_learning import (
    agent_suppression_script,
    read_page_snapshot,
    user_action_init_script,
    user_learning_enabled,
)
from src.browser.scroll_helpers import _GET_REMAINING_JS, auto_reveal_hidden_content

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_DEFAULT_TIMEOUT_MS = int(os.getenv("BROWSER_DEFAULT_TIMEOUT_MS", "90000"))
_SCREENSHOT_LOAD_TIMEOUT_MS = int(
    os.getenv("BROWSER_SCREENSHOT_LOAD_TIMEOUT_MS", "15000")
)
_SKIP_NETWORK_IDLE = os.getenv("BROWSER_SKIP_NETWORK_IDLE_WAIT", "true").lower() in (
    "1", "true", "yes", "on"
)
_FAST_DOM_ON_RETRY = os.getenv("BROWSER_STATE_FAST_DOM_ON_RETRY", "true").lower() in (
    "1", "true", "yes", "on"
)


def _exc_detail(exc: BaseException) -> str:
    """Playwright and some builtins expose empty str(e); keep logs actionable."""
    msg = str(exc).strip()
    if not msg:
        raw = getattr(exc, "message", None)
        if isinstance(raw, bytes):
            msg = raw.decode("utf-8", errors="replace")
        elif raw:
            msg = str(raw).strip()
    if not msg:
        msg = repr(exc)
    return f"{type(exc).__name__}: {msg}"


class CustomBrowserContext(BrowserContext):
    def __init__(
            self,
            browser: 'Browser',
            config: BrowserContextConfig | None = None,
            state: Optional[BrowserContextState] = None,
    ):
        super(CustomBrowserContext, self).__init__(browser=browser, config=config, state=state)
        self._last_page_fingerprint: Optional[str] = None
        self._timeouts_applied = False
        self._user_event_buffer: list[dict] = []
        self._user_learning_installed = False
        self._post_step_snapshot: Optional[dict] = None

    async def _initialize_session(self):
        session = await super()._initialize_session()
        if user_learning_enabled():
            await self.ensure_user_learning()
        return session

    async def _on_user_action(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        entry = dict(payload)
        entry["ts"] = entry.get("ts") or time.time()
        self._user_event_buffer.append(entry)
        if len(self._user_event_buffer) > 50:
            self._user_event_buffer = self._user_event_buffer[-50:]

    def drain_user_events(self) -> list[dict]:
        events = list(self._user_event_buffer)
        self._user_event_buffer.clear()
        return events

    async def capture_page_snapshot(self) -> dict:
        try:
            page = await self.get_agent_current_page()
            return await read_page_snapshot(page)
        except Exception as exc:
            logger.debug("capture_page_snapshot failed: %s", exc)
            return {}

    async def set_agent_action_suppression(self, ms: int = 2000) -> None:
        """Briefly ignore user-learning events while Playwright executes agent actions."""
        try:
            page = await self.get_agent_current_page()
            await page.evaluate(agent_suppression_script(ms))
        except Exception as exc:
            logger.debug("set_agent_action_suppression failed: %s", exc)

    async def ensure_user_learning(self) -> None:
        if self._user_learning_installed or not user_learning_enabled():
            return
        session = await self.get_session()
        context = session.context
        try:
            await context.expose_function("__buUserAction", self._on_user_action)
            await context.add_init_script(user_action_init_script())
            for page in context.pages:
                if page.is_closed():
                    continue
                if page.url.startswith(("chrome://", "chrome-extension://")):
                    continue
                try:
                    await page.evaluate(user_action_init_script())
                except Exception as exc:
                    logger.debug("User learning listener on %s: %s", page.url, exc)
            if not getattr(self, "_user_page_handler_set", False):
                context.on("page", self._attach_user_learning_to_page)
                self._user_page_handler_set = True
            self._user_learning_installed = True
            logger.info("User learning listeners installed (clicks/scrolls)")
        except Exception as exc:
            logger.warning("Could not install user learning listeners: %s", exc)

    async def _attach_user_learning_to_page(self, page: Page) -> None:
        if page.is_closed() or page.url.startswith(("chrome://", "chrome-extension://")):
            return
        try:
            await page.evaluate(user_action_init_script())
        except Exception as exc:
            logger.debug("User learning on new page %s: %s", page.url, exc)

    async def _ensure_playwright_timeouts(self) -> None:
        if self._timeouts_applied:
            return
        try:
            page = await self.get_agent_current_page()
            page.set_default_timeout(_DEFAULT_TIMEOUT_MS)
            page.set_default_navigation_timeout(_DEFAULT_TIMEOUT_MS)
            self._timeouts_applied = True
        except Exception as exc:
            logger.debug("Could not set page timeouts: %s", _exc_detail(exc))

    async def _wait_for_page_and_frames_load(self, timeout_overwrite: float | None = None):
        """LinkedIn/SPAs never go network-idle; skip that wait by default."""
        if _SKIP_NETWORK_IDLE:
            wait_s = timeout_overwrite or self.config.minimum_wait_page_load_time
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            return
        await super()._wait_for_page_and_frames_load(timeout_overwrite=timeout_overwrite)

    async def get_state(self, cache_clickable_elements_hashes: bool) -> BrowserState:
        await self._ensure_playwright_timeouts()
        return await super().get_state(
            cache_clickable_elements_hashes=cache_clickable_elements_hashes
        )

    async def _run_state_stage(self, stage: str, coro: Awaitable[_T]) -> _T:
        try:
            return await coro
        except Exception as exc:
            raise RuntimeError(f"{stage}: {_exc_detail(exc)}") from exc

    async def take_screenshot(self, full_page: bool = False) -> str:
        """
        Screenshot without blocking on full page 'load' (SPAs like LinkedIn rarely settle).
        """
        page = await self.get_agent_current_page()
        try:
            await page.wait_for_load_state(
                "domcontentloaded",
                timeout=_SCREENSHOT_LOAD_TIMEOUT_MS,
            )
        except Exception as exc:
            logger.debug(
                "Screenshot: domcontentloaded wait skipped: %s", _exc_detail(exc)
            )
        try:
            screenshot = await page.screenshot(
                full_page=full_page,
                animations="disabled",
                caret="initial",
                timeout=_SCREENSHOT_LOAD_TIMEOUT_MS,
            )
        except Exception as first_exc:
            logger.debug(
                "Screenshot retry without load wait: %s", _exc_detail(first_exc)
            )
            screenshot = await page.screenshot(
                full_page=full_page,
                animations="disabled",
                caret="initial",
                timeout=_SCREENSHOT_LOAD_TIMEOUT_MS,
            )
        return base64.b64encode(screenshot).decode("utf-8")

    async def _get_updated_state(self, focus_element: int = -1) -> BrowserState:
        """Retry transient DOM/screenshot failures; log full exception detail."""
        try:
            page = await self.get_agent_current_page()
            await page.evaluate("1")
        except Exception as exc:
            logger.debug("Current page is no longer accessible: %s", _exc_detail(exc))
            raise BrowserError("Browser closed: no valid pages available")

        max_retries = max(1, int(os.getenv("BROWSER_STATE_MAX_RETRIES", "3")))
        delay_s = float(os.getenv("BROWSER_STATE_RETRY_DELAY_S", "0.5"))
        last_err: Optional[Exception] = None

        await self._ensure_playwright_timeouts()

        for attempt in range(1, max_retries + 1):
            highlight = self.config.highlight_elements
            if attempt > 1 and _FAST_DOM_ON_RETRY:
                highlight = False
            try:
                await self._run_state_stage(
                    "remove_highlights", self.remove_highlights()
                )
                page = await self.get_agent_current_page()
                dom_service = DomService(page)
                content = await self._run_state_stage(
                    "dom_tree",
                    dom_service.get_clickable_elements(
                        focus_element=focus_element,
                        viewport_expansion=self.config.viewport_expansion,
                        highlight_elements=highlight,
                    ),
                )
                tabs_info = await self._run_state_stage(
                    "tabs", self.get_tabs_info()
                )
                screenshot_b64 = await self._run_state_stage(
                    "screenshot", self.take_screenshot()
                )
                pixels_above, pixels_below = await self._run_state_stage(
                    "scroll_metrics", self.get_scroll_info(page)
                )
                title = await self._run_state_stage(
                    "title",
                    asyncio.wait_for(page.title(), timeout=5.0),
                )

                self.current_state = BrowserState(
                    element_tree=content.element_tree,
                    selector_map=content.selector_map,
                    url=page.url,
                    title=title,
                    tabs=tabs_info,
                    screenshot=screenshot_b64,
                    pixels_above=pixels_above,
                    pixels_below=pixels_below,
                )
                return self.current_state
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    logger.warning(
                        "State update attempt %s/%s failed (%s), retrying…",
                        attempt,
                        max_retries,
                        _exc_detail(exc),
                    )
                    await asyncio.sleep(delay_s * attempt)
                    continue
                logger.error("Failed to update state: %s", _exc_detail(exc))
                if getattr(self, "current_state", None) is not None:
                    return self.current_state
                raise

        if last_err is not None:
            raise last_err
        raise RuntimeError("State update failed with no exception")

    async def get_scroll_info(self, page: Page) -> Tuple[int, int]:
        """Pixels above/below in the active scroll container (modal) or the window."""
        try:
            metrics = await page.evaluate(_GET_REMAINING_JS)
            if metrics:
                return int(metrics.get("above", 0)), int(metrics.get("below", 0))
        except Exception as exc:
            logger.debug("container scroll metrics failed, using window: %s", exc)
        return await super().get_scroll_info(page)

    async def build_page_briefing(
        self,
        state: BrowserState,
        *,
        include_footer_scan: bool = False,
        include_html_inspect: bool = True,
        deep_html: bool = False,
        enable_ocr: bool = True,
        use_fingerprint_cache: bool = True,
    ) -> Optional[str]:
        from src.browser.page_scrape_pipeline import build_briefing_from_scrape

        prev = self._last_page_fingerprint if use_fingerprint_cache else None
        briefing, fp = await build_briefing_from_scrape(
            self,
            state,
            include_footer_scan=include_footer_scan,
            include_html_inspect=include_html_inspect,
            deep_html=deep_html,
            enable_ocr=enable_ocr,
            previous_fingerprint=prev,
            skip_unchanged_full=use_fingerprint_cache,
        )
        if fp:
            self._last_page_fingerprint = fp
        return briefing

    def reset_page_fingerprint(self) -> None:
        self._last_page_fingerprint = None

    async def analyze_ui_context(self, state: BrowserState) -> Optional[str]:
        """Full page briefing (UI, form, OCR, errors, buttons)."""
        return await self.build_page_briefing(state, include_footer_scan=False)

    async def auto_reveal_if_needed(self, page: Page, threshold: int = 60) -> Optional[str]:
        return await auto_reveal_hidden_content(page, threshold=threshold)
