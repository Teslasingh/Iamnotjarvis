from __future__ import annotations

from typing import List, Optional

from friday.config import Settings, get_settings
from friday.llm.azure_openai import AzureOpenAILLM


class LLMClient:
    """Azure-only LLM client used by the agent."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._azure = AzureOpenAILLM(self.settings)

    def chat_with_tools(self, messages: List[dict], tools: List[dict]) -> dict:
        return self._azure.chat_with_tools(messages=messages, tools=tools)

    def chat(self, messages: List[dict], *, temperature: float = 0.2) -> str:
        return self._azure.chat(messages=messages, temperature=temperature)

