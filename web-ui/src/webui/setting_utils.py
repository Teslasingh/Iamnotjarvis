"""Helpers for reading/writing persisted WebUI settings."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.webui.webui_manager import WebuiManager


def coerce_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def init_value(manager: "WebuiManager", comp_id: str, default: Any) -> Any:
    """Component default from last saved UI settings, if present."""
    if comp_id in manager.saved_settings:
        return manager.saved_settings[comp_id]
    return default
