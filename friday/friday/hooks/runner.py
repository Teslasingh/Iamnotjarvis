from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from friday.hooks.registry import HookRegistry

logger = logging.getLogger(__name__)


class HookRunner:
    def __init__(self, registry: HookRegistry, timeout_seconds: int = 10) -> None:
        self.registry = registry
        self.timeout_seconds = timeout_seconds

    async def dispatch_gateway(self, event: Dict[str, Any]) -> None:
        for hook in self.registry.list_hooks():
            if hook.get("type") != "gateway":
                continue
            if not self._event_matches(hook, event):
                continue
            action = hook.get("action") or {}
            url = action.get("webhook_url")
            if url:
                try:
                    async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                        await client.post(url, json=event)
                except Exception as exc:
                    logger.warning("gateway hook failed: %s", exc)

    async def before_tool_call(
        self, tool: str, args: Dict[str, Any]
    ) -> Optional[str]:
        for hook in self.registry.list_hooks():
            if hook.get("type") != "plugin" or hook.get("hook") != "before_tool_call":
                continue
            match = hook.get("match") or {}
            if match.get("tool") and match.get("tool") != tool:
                continue
            action = hook.get("action") or {}
            pattern = action.get("deny_if_regex")
            if pattern and tool == "run_shell":
                cmd = str(args.get("command", ""))
                if re.search(pattern, cmd):
                    return f"blocked by hook {hook.get('id')}: command denied"
        return None

    @staticmethod
    def _event_matches(hook: Dict[str, Any], event: Dict[str, Any]) -> bool:
        events = hook.get("events") or []
        if events and event.get("type") not in events:
            return False
        filt = hook.get("filter") or {}
        for key, val in filt.items():
            if event.get(key) != val:
                return False
        return True
