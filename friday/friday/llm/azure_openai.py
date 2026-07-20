from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
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

    def _supports_custom_temperature(self) -> bool:
        """GPT-5 and o-series deployments only accept the default temperature (1)."""
        name = self.settings.azure_openai_deployment_name.lower()
        if "gpt-5" in name:
            return False
        for series in ("o1", "o3", "o4"):
            if name.startswith(series) or f"-{series}" in name:
                return False
        return True

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

    def chat_with_tools(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload: Dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._post(payload)
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return message, usage

    def chat(
        self, messages: List[Dict[str, Any]], *, temperature: float = 0.2
    ) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {"messages": messages}
        if self._supports_custom_temperature():
            payload["temperature"] = temperature
        data = self._post(payload)
        content = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return str(content).strip(), usage

    def vision(
        self,
        text: str,
        image_b64: str,
        mime_type: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a text prompt plus an inline base64 image to a vision-capable deployment."""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
        ]
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
        }
        if self._supports_custom_temperature():
            payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
        data = self._post(payload)
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return str(message.get("content") or "").strip(), usage
