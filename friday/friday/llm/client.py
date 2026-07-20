from __future__ import annotations

from typing import List, Optional

from friday.config import Settings, get_settings
from friday.llm.azure_openai import AzureOpenAILLM
from friday.llm.usage import TokenUsageStore


class LLMClient:
    """Azure-only LLM client used by the agent."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        usage_store: Optional[TokenUsageStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._azure = AzureOpenAILLM(self.settings)
        self.usage_store = usage_store

    def _record(self, source: str, usage: dict) -> None:
        if self.usage_store and self.settings.token_usage_enabled:
            self.usage_store.record(source, usage)

    def chat_with_tools(
        self,
        messages: List[dict],
        tools: List[dict],
        *,
        source: str = "agent_step",
    ) -> dict:
        message, usage = self._azure.chat_with_tools(messages=messages, tools=tools)
        self._record(source, usage)
        return message

    def chat(
        self,
        messages: List[dict],
        *,
        temperature: float = 0.2,
        source: str = "chat",
    ) -> str:
        content, usage = self._azure.chat(messages=messages, temperature=temperature)
        self._record(source, usage)
        return content
