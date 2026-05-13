from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import quote

import httpx

from friday.config import Settings


class AzureOpenAILLM:
    """Azure OpenAI chat-completions client."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        if not settings.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")
        if not settings.azure_openai_deployment_name:
            raise ValueError("DEPLOYMENT_NAME or AZURE_OPENAI_DEPLOYMENT_NAME is not set")

    def _url(self) -> str:
        deployment = quote(self.settings.azure_openai_deployment_name, safe="")
        return (
            f"{self.settings.azure_openai_endpoint}/openai/deployments/"
            f"{deployment}/chat/completions?api-version={self.settings.azure_openai_api_version}"
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "api-key": self.settings.azure_openai_api_key or "",
            "Content-Type": "application/json",
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(self._url(), headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    def complete_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": messages}
        return self._post(payload)["choices"][0]["message"]

    def chat_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._post(payload)["choices"][0]["message"]

    def chat(self, user_message: str, system: str = "") -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        return self.complete_messages(messages).get("content") or ""

    def invoke(self, prompt: str, system: str = "") -> str:
        return self.chat(prompt, system=system)

    def reset_conversation(self) -> None:
        # Azure calls are stateless in this application.
        return None
