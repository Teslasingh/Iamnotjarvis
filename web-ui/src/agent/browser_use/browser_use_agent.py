from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time

from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc
from browser_use.agent.views import (
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentStepInfo,
    ToolCallingMethod,
)
from browser_use.browser.views import BrowserStateHistory
from browser_use.utils import time_execution_async
from dotenv import load_dotenv
from browser_use.agent.message_manager.utils import is_model_without_tool_support
from langchain_core.messages import HumanMessage

from src.agent.goal_driven import RECOVERY_HINT_MESSAGE
from src.agent.session_context import JobSessionContext, session_inject_interval
from src.agent.user_learning import (
    dedup_learning_lines,
    detect_snapshot_delta,
    filter_agent_era_events,
    format_user_event,
    user_learning_enabled,
)
from src.agent.user_preferences import (
    UserPreferenceStore,
    UserPreferenceSynthesizer,
)
from src.browser.custom_context import CustomBrowserContext
from src.browser.page_scrape_pipeline import (
    DOM_CHANGING_ACTIONS,
    build_briefing_from_scrape,
)
from src.browser.search_scrape import (
    actions_trigger_search_settle,
    settle_page_after_search,
    should_run_full_search_scrape,
    wait_for_results_hint,
)

load_dotenv()
logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = (
        os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)

_STALL_REPEAT_THRESHOLD = 3


class BrowserUseAgent(Agent):
    def __init__(
        self,
        *args,
        goal_driven_autonomy: bool = True,
        enable_ocr: bool | None = None,
        job_application_mode: bool | None = None,
        enable_html_inspect: bool | None = None,
        job_session_context: JobSessionContext | None = None,
        learn_from_user_overrides: bool | None = None,
        user_preference_store: UserPreferenceStore | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._job_session: JobSessionContext | None = job_session_context
        if learn_from_user_overrides is None:
            learn_from_user_overrides = user_learning_enabled()
        self._learn_from_user_overrides = bool(learn_from_user_overrides)
        self._user_preferences: UserPreferenceStore | None = user_preference_store
        self._last_step_snapshot: dict | None = None
        self._last_step_actions: set[str] = set()
        self._step_started_at: float | None = None
        self._step_ended_at: float | None = None
        self._prev_step_window: tuple[float | None, float | None] = (None, None)
        self.goal_driven_autonomy = goal_driven_autonomy
        if enable_ocr is None:
            enable_ocr = os.getenv("OCR_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        self.enable_ocr = bool(enable_ocr)
        if job_application_mode is None:
            job_application_mode = os.getenv("JOB_APPLICATION_MODE", "true").lower() in (
                "1", "true", "yes", "on"
            )
        self.job_application_mode = bool(job_application_mode)
        if enable_html_inspect is None:
            enable_html_inspect = os.getenv("HTML_INSPECT_ENABLED", "true").lower() in (
                "1", "true", "yes", "on"
            )
        self.enable_html_inspect = bool(enable_html_inspect)
        self._recent_action_signatures: list[str] = []
        self._recovery_injected_this_step = False
        self._fast_rescrape = os.getenv("FAST_POST_ACTION_RESCRAPE", "false").lower() in (
            "1", "true", "yes", "on"
        )
        self._page_scrape_fast = os.getenv("PAGE_SCRAPE_FAST", "true").lower() in (
            "1", "true", "yes", "on"
        )
        self._pending_full_search_scrape = False
        self._preference_synthesizer = UserPreferenceSynthesizer(self.llm)

    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        tool_calling_method = self.settings.tool_calling_method
        if tool_calling_method == 'auto':
            if is_model_without_tool_support(self.model_name):
                return 'raw'
            elif self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    def _action_signature(self, model_output) -> str:
        if not model_output or not model_output.action:
            return ""
        try:
            payload = [a.model_dump(exclude_none=True) for a in model_output.action]
            raw = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            raw = str(model_output.action)
        goal = ""
        if model_output.current_state and model_output.current_state.next_goal:
            goal = model_output.current_state.next_goal
        return hashlib.sha256(f"{goal}|{raw}".encode()).hexdigest()[:16]

    async def _inject_page_briefing(
        self,
        ctx: CustomBrowserContext,
        *,
        footer_scan: bool = False,
        deep_html: bool = False,
        force_full: bool = False,
        state=None,
    ) -> None:
        """Scrape page (parallel tools) → single LLM briefing."""
        if state is None:
            state = await ctx.get_state(cache_clickable_elements_hashes=True)
        page = await ctx.get_current_page()
        below = int(getattr(state, "pixels_below", 0) or 0)
        above = int(getattr(state, "pixels_above", 0) or 0)
        scroll_nudge_threshold = 220 if self._page_scrape_fast else 180
        auto_scroll = os.getenv("AUTO_SCROLL_ON_BRIEFING", "false").lower() in (
            "1", "true", "yes", "on"
        )

        if auto_scroll and self.goal_driven_autonomy and below >= 200 and above < 60:
            auto_msg = await ctx.auto_reveal_if_needed(page, threshold=160)
            if auto_msg:
                ctx.reset_page_fingerprint()
                self._message_manager._add_message_with_tokens(
                    HumanMessage(
                        content=(
                            "[Auto-scroll] One gentle step down in centered modal.\n"
                            + auto_msg
                        )
                    )
                )
                logger.info("Auto-scroll: %s", auto_msg)
                state = await ctx.get_state(cache_clickable_elements_hashes=True)
                below = int(getattr(state, "pixels_below", 0) or 0)
                above = int(getattr(state, "pixels_above", 0) or 0)

        prev = None if force_full else ctx._last_page_fingerprint
        step_n = getattr(self.state, "n_steps", 0) or 0
        use_vision = bool(getattr(self.settings, "use_vision", False))
        url = getattr(state, "url", "") or ""
        task_text = getattr(self, "task", "") or ""
        full_search = should_run_full_search_scrape(
            url=url,
            pending=force_full,
            task=task_text,
            step_number=step_n,
        )
        briefing, fp = await build_briefing_from_scrape(
            ctx,
            state,
            include_footer_scan=footer_scan,
            include_html_inspect=self.enable_html_inspect,
            deep_html=deep_html or full_search,
            enable_ocr=self.enable_ocr,
            job_application_mode=self.job_application_mode,
            previous_fingerprint=prev,
            skip_unchanged_full=not (force_full or full_search),
            fast=self._page_scrape_fast and not deep_html and not full_search,
            step_number=step_n,
            force_ocr=force_full or full_search,
            use_vision=use_vision,
            force_full_scrape=full_search,
        )
        if fp:
            ctx._last_page_fingerprint = fp

        if briefing:
            scroll_nudge = ""
            if above >= 120:
                scroll_nudge = (
                    f"\n[Scroll] ~{above}px content above — use scroll_up or scroll_to_top "
                    "before scrolling down again.\n"
                )
            elif below >= scroll_nudge_threshold:
                scroll_nudge = (
                    f"\n[Scroll] ~{below}px below — use scroll_down in small steps or "
                    "scroll_element_into_view; avoid scroll_to_bottom until review/submit.\n"
                )
            self._message_manager._add_message_with_tokens(
                HumanMessage(content=briefing + scroll_nudge)
            )

    async def _resolve_briefing_state(self, ctx: CustomBrowserContext):
        """Fresh state after search/navigation; otherwise reuse cache until super().step()."""
        full_search = self._pending_full_search_scrape
        if full_search:
            page = await ctx.get_current_page()
            await settle_page_after_search(page)
            hint = await wait_for_results_hint(page)
            if hint:
                logger.info("Search results selector visible: %s", hint)
            ctx.reset_page_fingerprint()
            state = await ctx.get_state(cache_clickable_elements_hashes=True)
            self._pending_full_search_scrape = False
            return state, True

        session = await ctx.get_session()
        state = session.cached_state or getattr(ctx, "current_state", None)
        if state is None:
            state = await ctx.get_state(cache_clickable_elements_hashes=True)
        url = getattr(state, "url", "") or ""
        task_text = getattr(self, "task", "") or ""
        if should_run_full_search_scrape(
            url=url,
            task=task_text,
            step_number=getattr(self.state, "n_steps", 0) or 0,
        ):
            page = await ctx.get_current_page()
            await settle_page_after_search(page, extra_wait_s=1.0)
            ctx.reset_page_fingerprint()
            state = await ctx.get_state(cache_clickable_elements_hashes=True)
            return state, True
        return state, False

    async def _inject_ui_context(self) -> None:
        if not self.goal_driven_autonomy and not self.enable_ocr and not self.enable_html_inspect:
            return
        ctx = self.browser_context
        if not isinstance(ctx, CustomBrowserContext):
            return
        try:
            state, full_search = await self._resolve_briefing_state(ctx)
            if not self.goal_driven_autonomy:
                await self._inject_page_briefing(
                    ctx,
                    footer_scan=False,
                    deep_html=full_search,
                    force_full=full_search,
                    state=state,
                )
                return

            step_n = self.state.n_steps
            footer_scan = False
            deep_interval = 15 if self._page_scrape_fast else 10
            deep_html = full_search or (
                self.job_application_mode and step_n > 0 and step_n % deep_interval == 0
            )
            await self._inject_page_briefing(
                ctx,
                footer_scan=footer_scan,
                deep_html=deep_html,
                force_full=full_search,
                state=state,
            )
        except Exception as exc:
            logger.debug("UI context injection skipped: %s", exc)

    def _action_names(self, action_model) -> set[str]:
        try:
            return {
                k
                for k, v in action_model.model_dump(exclude_unset=True).items()
                if v is not None
            }
        except Exception:
            return set()

    async def _fast_rescrape_after_dom_change(self) -> None:
        """Invalidate cache so the next step runs a fresh scrape (no duplicate full scrape)."""
        ctx = self.browser_context
        if not isinstance(ctx, CustomBrowserContext):
            return
        ctx.reset_page_fingerprint()
        self._message_manager._add_message_with_tokens(
            HumanMessage(
                content="[Page updated] Use fresh element indexes on the next step."
            )
        )

    def _capture_session_from_last_step(self) -> None:
        if not self._job_session:
            return
        history = self.state.history.history
        if not history:
            return
        item = history[-1]
        url = title = memory = evaluation = next_goal = ""
        snippets: list[str] = []
        if item.model_output and item.model_output.current_state:
            cs = item.model_output.current_state
            memory = cs.memory or ""
            evaluation = cs.evaluation_previous_goal or ""
            next_goal = cs.next_goal or ""
        if item.state:
            url = getattr(item.state, "url", "") or ""
            title = getattr(item.state, "title", "") or ""
        if item.result:
            for r in item.result:
                if r.extracted_content:
                    snippets.append(str(r.extracted_content)[:200])
        self._job_session.update_from_step(
            url=url,
            title=title,
            memory=memory,
            evaluation=evaluation,
            next_goal=next_goal,
            result_snippets=snippets,
        )
        self._job_session.save()

    def _maybe_inject_recovery_hint(self, model_output) -> None:
        if not self.goal_driven_autonomy or self._recovery_injected_this_step:
            return
        sig = self._action_signature(model_output)
        if not sig:
            return
        self._recent_action_signatures.append(sig)
        if len(self._recent_action_signatures) > _STALL_REPEAT_THRESHOLD:
            self._recent_action_signatures.pop(0)
        if (
            len(self._recent_action_signatures) >= _STALL_REPEAT_THRESHOLD
            and len(set(self._recent_action_signatures)) == 1
        ):
            self._message_manager._add_message_with_tokens(
                HumanMessage(content=RECOVERY_HINT_MESSAGE)
            )
            self._recovery_injected_this_step = True
            self._recent_action_signatures.clear()

    async def multi_act(
        self,
        actions: list,
        check_for_new_elements: bool = True,
    ) -> list[ActionResult]:
        """Execute actions; fast re-scrape when page/options change."""
        if self._learn_from_user_overrides:
            ctx = self.browser_context
            if isinstance(ctx, CustomBrowserContext):
                await ctx.set_agent_action_suppression()
        results = await super().multi_act(actions, check_for_new_elements)

        if not self._fast_rescrape or not self.goal_driven_autonomy:
            return results

        dom_changed = False
        for r in results:
            if r.extracted_content and (
                "new appeared" in (r.extracted_content or "").lower()
                or "index changed" in (r.extracted_content or "").lower()
                or "page changed" in (r.extracted_content or "").lower()
            ):
                dom_changed = True
                break

        if not dom_changed:
            for action in actions[: len(results)]:
                names = self._action_names(action)
                if names & DOM_CHANGING_ACTIONS:
                    dom_changed = True
                    break

        if dom_changed:
            action_names: set[str] = set()
            for action in actions[: len(results)]:
                action_names |= self._action_names(action)
            if actions_trigger_search_settle(action_names):
                self._pending_full_search_scrape = True
                ctx = self.browser_context
                if isinstance(ctx, CustomBrowserContext):
                    page = await ctx.get_current_page()
                    await settle_page_after_search(page)
                    await wait_for_results_hint(page)
                    ctx.reset_page_fingerprint()
                self._message_manager._add_message_with_tokens(
                    HumanMessage(
                        content=(
                            "[Search/navigation] Page settled — next step will run full "
                            "DOM + OCR + form briefing. Read job cards from indexes and OCR."
                        )
                    )
                )
            else:
                await self._fast_rescrape_after_dom_change()

        return results

    def _collect_last_step_actions(self) -> set[str]:
        history = self.state.history.history
        if not history:
            return set()
        item = history[-1]
        names: set[str] = set()
        if item.model_output and item.model_output.action:
            for action in item.model_output.action:
                names |= self._action_names(action)
        return names

    def _inject_user_preferences(self) -> None:
        if not self._learn_from_user_overrides or not self._user_preferences:
            return
        block = self._user_preferences.format_injection()
        if block:
            self._message_manager._add_message_with_tokens(HumanMessage(content=block))

    async def _maybe_synthesize_preferences(self) -> None:
        if not self._learn_from_user_overrides or not self._user_preferences:
            return
        self._preference_synthesizer.set_llm(getattr(self, "llm", None))
        if await self._preference_synthesizer.synthesize(self._user_preferences):
            self._inject_user_preferences()

    async def finalize_user_preferences(self) -> None:
        """Run a final synthesis pass when a task ends."""
        if not self._learn_from_user_overrides or not self._user_preferences:
            return
        self._preference_synthesizer.set_llm(getattr(self, "llm", None))
        await self._preference_synthesizer.synthesize_remaining(self._user_preferences)

    async def _inject_user_learnings(self) -> None:
        if not self._learn_from_user_overrides:
            return
        self._inject_user_preferences()
        ctx = self.browser_context
        if not isinstance(ctx, CustomBrowserContext):
            return
        try:
            await ctx.ensure_user_learning()
        except Exception as exc:
            logger.debug("ensure_user_learning skipped: %s", exc)

        step_start, step_end = self._prev_step_window
        events = filter_agent_era_events(
            ctx.drain_user_events(),
            self._last_step_actions,
            step_start,
            step_end,
        )

        lines: list[str] = []
        for event in events:
            line = format_user_event(event)
            if not line:
                continue
            lines.append(line)
            if self._job_session:
                self._job_session.record_user_learning(
                    line,
                    url=str(event.get("url") or ""),
                    learning_type=str(event.get("type") or ""),
                )
            if self._user_preferences:
                self._user_preferences.record_raw(line)

        current = await ctx.capture_page_snapshot()
        for line in detect_snapshot_delta(
            self._last_step_snapshot, current, self._last_step_actions
        ):
            lines.append(line)
            if self._job_session:
                self._job_session.record_user_learning(
                    line,
                    url=str(current.get("url") or ""),
                    learning_type="delta",
                )
            if self._user_preferences:
                self._user_preferences.record_raw(line)

        lines = dedup_learning_lines(lines)
        if self._user_preferences and lines:
            self._user_preferences.save()

        if lines:
            block = (
                "[User corrections — the user manually adjusted the browser; treat as ground truth]\n"
                + "\n".join(f"- {line}" for line in lines)
                + "\n- Align your next goal and actions with what the user did."
            )
            self._message_manager._add_message_with_tokens(HumanMessage(content=block))
            if self._job_session:
                self._job_session.save()
            logger.info("Injected %d user learning(s) for next step", len(lines))

        await self._maybe_synthesize_preferences()

    async def _update_step_snapshot(self) -> None:
        if not self._learn_from_user_overrides:
            return
        ctx = self.browser_context
        if isinstance(ctx, CustomBrowserContext):
            self._last_step_snapshot = await ctx.capture_page_snapshot()
        self._last_step_actions = self._collect_last_step_actions()

    def _inject_session_context(self) -> None:
        if not self._job_session:
            return
        step_n = getattr(self.state, "n_steps", 0) or 0
        block = self._job_session.format_injection(
            step_number=step_n,
            inject_interval=session_inject_interval(),
        )
        if block:
            self._message_manager._add_message_with_tokens(
                HumanMessage(content=block)
            )

    @time_execution_async('--step (agent)')
    async def step(self, step_info: AgentStepInfo | None = None) -> None:
        self._recovery_injected_this_step = False
        await self._inject_user_learnings()
        self._inject_session_context()
        await self._inject_ui_context()
        if self._learn_from_user_overrides:
            self._step_started_at = time.time()
        await super().step(step_info)
        if self._learn_from_user_overrides:
            self._step_ended_at = time.time()
            self._prev_step_window = (self._step_started_at, self._step_ended_at)
        self._capture_session_from_last_step()
        await self._update_step_snapshot()
        history = self.state.history.history
        if history and history[-1].model_output:
            self._maybe_inject_recovery_hint(history[-1].model_output)

    @time_execution_async('--run (agent)')
    async def run(
            self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None,
            on_step_end: AgentHookFunc | None = None
    ) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""

        loop = asyncio.get_event_loop()

        from browser_use.utils import SignalHandler

        signal_handler = SignalHandler(
            loop=loop,
            pause_callback=self.pause,
            resume_callback=self.resume,
            custom_exit_callback=None,
            exit_on_second_int=True,
        )
        signal_handler.register()

        try:
            self._log_agent_run()

            if self.initial_actions:
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            for step in range(max_steps):
                if self.state.paused:
                    signal_handler.wait_for_resume()
                    signal_handler.reset()

                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    break

                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)
                    if self.state.stopped:
                        break

                if on_step_start is not None:
                    await on_step_start(self)

                step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                await self.step(step_info)

                if on_step_end is not None:
                    await on_step_end(self)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    break
            else:
                error_message = 'Failed to complete task in maximum steps'

                self.state.history.history.append(
                    AgentHistory(
                        model_output=None,
                        result=[ActionResult(error=error_message, include_in_memory=True)],
                        state=BrowserStateHistory(
                            url='',
                            title='',
                            tabs=[],
                            interacted_element=[],
                            screenshot=None,
                        ),
                        metadata=None,
                    )
                )

                logger.info(f'❌ {error_message}')

            return self.state.history

        except KeyboardInterrupt:
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history

        finally:
            signal_handler.unregister()

            if self.settings.save_playwright_script_path:
                logger.info(
                    f'Agent run finished. Attempting to save Playwright script to: {self.settings.save_playwright_script_path}'
                )
                try:
                    keys = list(self.sensitive_data.keys()) if self.sensitive_data else None
                    self.state.history.save_as_playwright_script(
                        self.settings.save_playwright_script_path,
                        sensitive_data_keys=keys,
                        browser_config=self.browser.config,
                        context_config=self.browser_context.config,
                    )
                except Exception as script_gen_err:
                    logger.error(f'Failed to save Playwright script: {script_gen_err}', exc_info=True)

            await self.close()

            if self.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif

                create_history_gif(task=self.task, history=self.state.history, output_path=output_path)
