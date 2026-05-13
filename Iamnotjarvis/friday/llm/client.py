from __future__ import annotations

from typing import Any, Dict, List, Optional

from friday.config import Settings, get_settings
from friday.llm.azure_openai import AzureOpenAILLM


class LLMClient:
    """Azure-only LLM client used by the agent and legacy codegen."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._azure = AzureOpenAILLM(self.settings)

    def chat_with_tools(self, messages: List[dict], tools: List[dict]) -> dict:
        return self._azure.chat_with_tools(messages=messages, tools=tools)

    def chat(self, prompt: str, system: str = "") -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._azure.complete_messages(messages).get("content", "") or ""

    def invoke(self, prompt: str, system: str = "") -> str:
        return self.chat(prompt, system=system)

    def reset_conversation(self) -> None:
        return None
