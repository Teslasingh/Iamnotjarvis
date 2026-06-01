from __future__ import annotations

from typing import Any, Dict, List


def to_sharegpt(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    conversations: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content") or "")
        if role == "user":
            conversations.append({"from": "human", "value": content})
        elif role == "assistant":
            conversations.append({"from": "gpt", "value": content})
        elif role == "tool":
            conversations.append({"from": "tool", "value": content})
        elif role == "system":
            conversations.append({"from": "system", "value": content})
    return {"conversations": conversations}
