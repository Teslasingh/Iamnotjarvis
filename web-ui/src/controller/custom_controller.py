import base64
import pyperclip
from typing import Optional, Type, Callable, Dict, Any, Union, Awaitable, TypeVar
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from browser_use.controller.registry.service import Registry, RegisteredAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging
import inspect
import asyncio
import os
from langchain_core.language_models.chat_models import BaseChatModel
from browser_use.agent.views import ActionModel, ActionResult

from src.utils.mcp_client import create_tool_param_model, setup_mcp_client_and_tools
from src.browser.dropdown_helpers import (
    get_dropdown_options_robust,
    select_dropdown_option_robust,
)
from src.browser.ocr_service import is_ocr_available, search_visible_text
from src.browser.html_inspector import run_deep_html_inspect
from src.browser.page_understanding import (
    discover_footer_controls as discover_footer_controls_scan,
    run_full_page_scan,
)
from src.browser.search_scrape import settle_page_after_search, wait_for_results_hint
from src.browser.scroll_helpers import (
    auto_reveal_hidden_content,
    format_scroll_result,
    scroll_aggressive,
    scroll_by,
    scroll_to_edge,
)

from browser_use.utils import time_execution_sync

# Replace library actions with robust web-ui implementations.
_SCROLL_ACTION_EXCLUDES = ["scroll_down", "scroll_up"]
_DROPDOWN_ACTION_EXCLUDES = ["get_dropdown_options", "select_dropdown_option"]
_ACTION_EXCLUDES = [*_SCROLL_ACTION_EXCLUDES, *_DROPDOWN_ACTION_EXCLUDES]

logger = logging.getLogger(__name__)

Context = TypeVar('Context')


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None,
                 ask_assistant_callback: Optional[Union[Callable[[str, BrowserContext], Dict[str, Any]], Callable[
                     [str, BrowserContext], Awaitable[Dict[str, Any]]]]] = None,
                 ):
        merged_excludes = list(dict.fromkeys([*exclude_actions, *_ACTION_EXCLUDES]))
        super().__init__(exclude_actions=merged_excludes, output_model=output_model)
        for action_name in _ACTION_EXCLUDES:
            if action_name in self.registry.exclude_actions:
                self.registry.exclude_actions.remove(action_name)
        self._register_custom_actions()
        self._register_enhanced_scroll_actions()
        self._register_robust_dropdown_actions()
        self.ask_assistant_callback = ask_assistant_callback
        self.mcp_client = None
        self.mcp_server_config = None

    def _register_enhanced_scroll_actions(self):
        """Scroll modals/panels first, then the page — required for job apps and wizards."""

        @self.registry.action(
            "Scroll DOWN one moderate step in the centered popup/modal (not the background page). "
            "Use when a field or button is slightly below the fold. Optional amount: pixels (~35% viewport default). "
            "Prefer scroll_element_into_view on an index when you know the target.",
            param_model=ScrollAction,
        )
        async def scroll_down(params: ScrollAction, browser: BrowserContext):
            page = await browser.get_current_page()
            result = await scroll_by(page, "down", params.amount)
            if abs(result.get("moved", 0) or 0) < 5:
                result = await scroll_aggressive(page, "down", max_attempts=1)
            msg = format_scroll_result(result, "Scrolled down")
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Scroll UP. Scrolls inside the active modal/dialog/panel when present, else the page. "
            "Optional amount: pixels; default ~one viewport.",
            param_model=ScrollAction,
        )
        async def scroll_up(params: ScrollAction, browser: BrowserContext):
            page = await browser.get_current_page()
            result = await scroll_by(page, "up", params.amount)
            msg = format_scroll_result(result, "Scrolled up")
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Scroll toward the BOTTOM in a few small steps (not one jump). Use on review/submit screens "
            "when Submit/Next is off-screen — not while filling fields at the top of a form."
        )
        async def scroll_to_bottom(browser: BrowserContext):
            page = await browser.get_current_page()
            result = await scroll_to_edge(page, "bottom")
            await asyncio.sleep(0.15)
            msg = format_scroll_result(result, "Scrolled to bottom")
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Scroll to the TOP of the modal/dialog/panel or page."
        )
        async def scroll_to_top(browser: BrowserContext):
            page = await browser.get_current_page()
            result = await scroll_to_edge(page, "top")
            await asyncio.sleep(0.35)
            msg = format_scroll_result(result, "Scrolled to top")
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Reveal a little more content below in the centered popup/modal (one small scroll step)."
        )
        async def reveal_more_content(browser: BrowserContext):
            page = await browser.get_current_page()
            auto_msg = await auto_reveal_hidden_content(page, threshold=120)
            if auto_msg:
                logger.info(auto_msg)
                return ActionResult(extracted_content=auto_msg, include_in_memory=True)
            result = await scroll_by(page, "down", None)
            msg = format_scroll_result(result, "Revealed content (scroll)")
            logger.info(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)

    def _register_robust_dropdown_actions(self):
        @self.registry.action(
            "List options for a dropdown at index. Works for native <select> and custom comboboxes "
            "(opens the menu if needed). Call before select_dropdown_option when unsure of exact labels."
        )
        async def get_dropdown_options(index: int, browser: BrowserContext):
            return await get_dropdown_options_robust(browser, index)

        @self.registry.action(
            "Select a dropdown option at index by visible text. Supports native <select> (fuzzy match), "
            "nested selects, and custom comboboxes (click + option, type-ahead). "
            "Use get_dropdown_options first for required fields. Partial text OK (e.g. '60 days' vs '60')."
        )
        async def select_dropdown_option(index: int, text: str, browser: BrowserContext):
            return await select_dropdown_option_robust(browser, index, text)

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action(
            "Deep HTML/DOM inspection: all forms, labels, values, required fields, dialogs, "
            "buttons, headings, readable text. Use when indexes are unclear or to verify filled fields."
        )
        async def inspect_page_html(browser: BrowserContext):
            page = await browser.get_current_page()
            msg = await run_deep_html_inspect(page)
            logger.info("inspect_page_html completed (%s chars)", len(msg))
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Full page scan for job applications: DOM fields, validation errors, OCR, buttons, "
            "scroll to bottom to find Submit/Next. Use when stuck or on review screen (LinkedIn)."
        )
        async def scan_application_page(browser: BrowserContext):
            msg = await run_full_page_scan(browser)
            logger.info("scan_application_page completed")
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Scroll to bottom of centered modal/page and list footer buttons (Submit, Next, Review). "
            "Use before submitting LinkedIn Easy Apply or when buttons are off-screen."
        )
        async def discover_footer_controls(browser: BrowserContext):
            msg = await discover_footer_controls_scan(browser)
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "When the form section looks complete (fields filled, no required empty inputs), "
            "click Next/Continue/Submit. Uses DOM field probe + OCR. Call instead of re-filling inputs."
        )
        async def advance_form_step(browser: BrowserContext):
            from src.browser.form_progress import build_form_progress_message, probe_form_fields
            from src.browser.ocr_service import is_ocr_available
            import base64 as b64mod

            page = await browser.get_current_page()
            state = await browser.get_state(cache_clickable_elements_hashes=False)
            dom_probe = await probe_form_fields(page)
            ocr_texts: list = []
            if is_ocr_available():
                try:
                    from src.browser.ocr_service import get_ocr_lines_from_screenshot_b64, ocr_lines_as_text_list
                    png = await page.screenshot(type="png", full_page=False)
                    lines = await get_ocr_lines_from_screenshot_b64(
                        b64mod.b64encode(png).decode("utf-8")
                    )
                    ocr_texts = ocr_lines_as_text_list(lines)
                except Exception:
                    pass
            msg = build_form_progress_message(
                state, dom_probe=dom_probe, ocr_line_texts=ocr_texts
            )
            req_empty = dom_probe.get("required_empty") or []
            if req_empty:
                labels = ", ".join(f.get("label", "?") for f in req_empty[:5])
                return ActionResult(
                    extracted_content=(
                        (msg or "") + f"\nCannot advance yet — required empty: {labels}. Fill these first."
                    ),
                    include_in_memory=True,
                )
            # Try to click Next/Submit via OCR-matched element in tree
            elements_text = ""
            try:
                elements_text = state.element_tree.clickable_elements_to_string()
            except Exception:
                pass
            import re
            for line in elements_text.split("\n"):
                if re.search(r"\b(next|continue|submit|review)\b", line, re.I):
                    m = re.search(r"\[(\d+)\]", line)
                    if m:
                        idx = int(m.group(1))
                        result = await self.registry.execute_action(
                            "click_element_by_index",
                            {"index": idx},
                            browser=browser,
                        )
                        if isinstance(result, ActionResult) and not result.error:
                            return ActionResult(
                                extracted_content=f"Advanced form: clicked [{idx}] ({line.strip()[:50]})",
                                include_in_memory=True,
                            )
                        if isinstance(result, ActionResult):
                            return result
            return ActionResult(
                extracted_content=(
                    (msg or "Section may be complete.")
                    + "\nUse click_element_by_index on Next/Continue/Submit from DOM or OCR hints."
                ),
                include_in_memory=True,
            )

        @self.registry.action(
            "Wait for search or job-listing results to load (network + DOM settle), then refresh understanding. "
            "Use after submitting a search or opening a jobs page before clicking listings."
        )
        async def wait_for_search_results(browser: BrowserContext):
            page = await browser.get_current_page()
            await settle_page_after_search(page)
            sel = await wait_for_results_hint(page)
            msg = "[Search settle] Waited for page load."
            if sel:
                msg += f" Results region detected ({sel})."
            msg += " Next step includes full DOM + OCR briefing."
            return ActionResult(extracted_content=msg, include_in_memory=True)

        @self.registry.action(
            "Search for visible text on the current screen using OCR (pytesseract). "
            "Use when DOM indexes miss labels, job titles, Submit/Next buttons, or search box hints. "
            "Then map findings to click_element_by_index or scroll_to_text."
        )
        async def ocr_search_visible_text(query: str, browser: BrowserContext):
            if not is_ocr_available():
                return ActionResult(
                    extracted_content=(
                        "OCR unavailable. Install Tesseract OCR and set TESSERACT_CMD in .env if needed."
                    ),
                    include_in_memory=True,
                )
            page = await browser.get_current_page()
            try:
                png = await page.screenshot(type="png", full_page=False)
                b64 = base64.b64encode(png).decode("utf-8")
                msg = await search_visible_text(b64, query)
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f"ocr_search_visible_text failed: {e}")

        @self.registry.action(
            "When executing tasks, prioritize autonomous completion. However, if you encounter a definitive blocker "
            "that prevents you from proceeding independently – such as needing credentials you don't possess, "
            "requiring subjective human judgment, needing a physical action performed, encountering complex CAPTCHAs, "
            "or facing limitations in your capabilities – you must request human assistance."
        )
        async def ask_for_assistant(query: str, browser: BrowserContext):
            if self.ask_assistant_callback:
                if inspect.iscoroutinefunction(self.ask_assistant_callback):
                    user_response = await self.ask_assistant_callback(query, browser)
                else:
                    user_response = self.ask_assistant_callback(query, browser)
                msg = f"AI ask: {query}. User response: {user_response['response']}"
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            else:
                return ActionResult(extracted_content="Human cannot help you. Please try another way.",
                                    include_in_memory=True)

        @self.registry.action(
            "Scroll the element at the given index into the visible viewport. "
            "Use when controls are inside a modal, drawer, or long form and are not yet interactable."
        )
        async def scroll_element_into_view(index: int, browser: BrowserContext):
            dom_el = await browser.get_dom_element_by_index(index)
            if dom_el is None:
                return ActionResult(error=f'No element at index {index}')
            el = await browser.get_locate_element(dom_el)
            if el is None:
                return ActionResult(error=f'Could not locate element at index {index}')
            try:
                await el.scroll_into_view_if_needed()
                await asyncio.sleep(0.4)
                msg = f'Scrolled element [{index}] into view'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=f'scroll_element_into_view failed: {e}')

        @self.registry.action(
            'Upload file to interactive element with file path ',
        )
        async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
            if path not in available_file_paths:
                return ActionResult(error=f'File path {path} is not available')

            if not os.path.exists(path):
                return ActionResult(error=f'File {path} does not exist')

            dom_el = await browser.get_dom_element_by_index(index)

            file_upload_dom_el = dom_el.get_file_upload_element()

            if file_upload_dom_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            file_upload_el = await browser.get_locate_element(file_upload_dom_el)

            if file_upload_el is None:
                msg = f'No file upload element found at index {index}'
                logger.info(msg)
                return ActionResult(error=msg)

            try:
                await file_upload_el.set_input_files(path)
                msg = f'Successfully uploaded file to index {index}'
                logger.info(msg)
                return ActionResult(extracted_content=msg, include_in_memory=True)
            except Exception as e:
                msg = f'Failed to upload file to index {index}: {str(e)}'
                logger.info(msg)
                return ActionResult(error=msg)

    @time_execution_sync('--act')
    async def act(
            self,
            action: ActionModel,
            browser_context: Optional[BrowserContext] = None,
            #
            page_extraction_llm: Optional[BaseChatModel] = None,
            sensitive_data: Optional[Dict[str, str]] = None,
            available_file_paths: Optional[list[str]] = None,
            #
            context: Context | None = None,
    ) -> ActionResult:
        """Execute an action"""

        try:
            for action_name, params in action.model_dump(exclude_unset=True).items():
                if params is not None:
                    if action_name.startswith("mcp"):
                        # this is a mcp tool
                        logger.debug(f"Invoke MCP tool: {action_name}")
                        mcp_tool = self.registry.registry.actions.get(action_name).function
                        result = await mcp_tool.ainvoke(params)
                    else:
                        result = await self.registry.execute_action(
                            action_name,
                            params,
                            browser=browser_context,
                            page_extraction_llm=page_extraction_llm,
                            sensitive_data=sensitive_data,
                            available_file_paths=available_file_paths,
                            context=context,
                        )

                    if isinstance(result, str):
                        return ActionResult(extracted_content=result)
                    elif isinstance(result, ActionResult):
                        return result
                    elif result is None:
                        return ActionResult()
                    else:
                        raise ValueError(f'Invalid action result type: {type(result)} of {result}')
            return ActionResult()
        except Exception as e:
            raise e

    async def setup_mcp_client(self, mcp_server_config: Optional[Dict[str, Any]] = None):
        self.mcp_server_config = mcp_server_config
        if self.mcp_server_config:
            self.mcp_client = await setup_mcp_client_and_tools(self.mcp_server_config)
            self.register_mcp_tools()

    def register_mcp_tools(self):
        """
        Register the MCP tools used by this controller.
        """
        if self.mcp_client:
            for server_name in self.mcp_client.server_name_to_tools:
                for tool in self.mcp_client.server_name_to_tools[server_name]:
                    tool_name = f"mcp.{server_name}.{tool.name}"
                    self.registry.registry.actions[tool_name] = RegisteredAction(
                        name=tool_name,
                        description=tool.description,
                        function=tool,
                        param_model=create_tool_param_model(tool),
                    )
                    logger.info(f"Add mcp tool: {tool_name}")
                logger.debug(
                    f"Registered {len(self.mcp_client.server_name_to_tools[server_name])} mcp tools for {server_name}")
        else:
            logger.warning(f"MCP client not started.")

    async def close_mcp_client(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
