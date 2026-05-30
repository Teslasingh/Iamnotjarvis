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
        with httpx.Client(timeout=float(self.settings.llm_timeout_seconds)) as client:
            resp = client.post(self._url(), headers=self._headers(), json=payload)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = resp.text[:2000]
                raise RuntimeError(
                    f"Azure OpenAI HTTP {resp.status_code}: {body}"
                ) from exc
            return resp.json()

    def chat_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return self._post(payload)["choices"][0]["message"]

    def chat(self, messages: List[Dict[str, Any]], *, temperature: float = 0.2) -> str:
        payload: Dict[str, Any] = {"messages": messages, "temperature": temperature}
        content = self._post(payload)["choices"][0]["message"].get("content") or ""
        return str(content).strip()

