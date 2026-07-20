"""Shared OpenAI / Azure OpenAI client with retry and JSON parsing."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AzureOpenAI, OpenAI

from config import settings
from retry import with_retry

logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        return AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
    kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    attempts: int = 3,
) -> str:
    """Create a chat completion with retries on transient failures."""
    if not settings.llm_configured:
        raise RuntimeError("LLM is not configured")

    client = get_openai_client()

    def _call() -> str:
        response = client.chat.completions.create(
            model=settings.llm_model,
            **settings.chat_temperature(temperature),
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    return with_retry(_call, attempts=attempts, operation_name="llm.chat_completion")


def chat_completion_json(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    attempts: int = 3,
) -> dict[str, Any]:
    content = chat_completion(messages, temperature=temperature, attempts=attempts)
    return json_from_text(content)


def json_from_text(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON: %s", exc)
        raise
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the model")
    return data


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
    return []
