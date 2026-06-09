"""Auto-save / restore Gradio UI settings across restarts."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from src.webui.webui_manager import WebuiManager

logger = logging.getLogger(__name__)

LAST_SETTINGS_FILENAME = "last_ui_settings.json"

_SKIP_ID_SUFFIXES = (
    ".chatbot",
    ".browser_view",
    ".recording_gif",
    ".agent_history_file",
    ".markdown_display",
    ".markdown_download",
    ".run_button",
    ".stop_button",
    ".start_button",
    ".clear_button",
    ".pause_resume_button",
    ".reload_prompt_button",
    ".load_config_button",
    ".save_config_button",
    ".config_status",
    ".config_file",
)

_SKIP_COMPONENT_TYPES = (
    gr.Button,
    gr.File,
    gr.Chatbot,
    gr.Markdown,
    gr.HTML,
    gr.Image,
)


def last_settings_path(manager: "WebuiManager") -> str:
    return os.path.join(manager.settings_save_dir, LAST_SETTINGS_FILENAME)


def load_saved_settings(manager: "WebuiManager") -> Dict[str, Any]:
    path = last_settings_path(manager)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            logger.info("Loaded UI settings from %s (%s keys)", path, len(data))
            return data
    except Exception as exc:
        logger.warning("Could not load UI settings from %s: %s", path, exc)
    return {}


def write_saved_settings(manager: "WebuiManager", settings: Dict[str, Any]) -> None:
    path = last_settings_path(manager)
    os.makedirs(manager.settings_save_dir, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2, ensure_ascii=False)
        manager.saved_settings = dict(settings)
        logger.debug("Persisted UI settings (%s keys) to %s", len(settings), path)
    except Exception as exc:
        logger.warning("Could not persist UI settings: %s", exc)


def is_persistable(comp: "Component", comp_id: str) -> bool:
    if any(comp_id.endswith(suffix) for suffix in _SKIP_ID_SUFFIXES):
        return False
    if comp_id.startswith("load_save_config."):
        return False
    if isinstance(comp, _SKIP_COMPONENT_TYPES):
        return False
    if str(getattr(comp, "interactive", True)).lower() == "false":
        return False
    return True


def collect_settings(
    manager: "WebuiManager", components: Dict["Component", Any]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for comp, value in components.items():
        if comp not in manager.component_to_id:
            continue
        comp_id = manager.component_to_id[comp]
        if not is_persistable(comp, comp_id):
            continue
        out[comp_id] = value
    return out


def persist_from_components(
    manager: "WebuiManager", components: Dict["Component", Any]
) -> None:
    write_saved_settings(manager, collect_settings(manager, components))


def list_persistable_components(manager: "WebuiManager") -> List["Component"]:
    items = []
    for comp_id, comp in manager.id_to_component.items():
        if is_persistable(comp, comp_id):
            items.append((comp_id, comp))
    items.sort(key=lambda x: x[0])
    return [comp for _, comp in items]


def register_settings_persistence(manager: "WebuiManager", demo: gr.Blocks) -> None:
    """Auto-save on change; restore on page load."""
    persistable = list_persistable_components(manager)
    if not persistable:
        return

    def _autosave(*values: Any) -> None:
        mapping = {comp: val for comp, val in zip(persistable, values)}
        persist_from_components(manager, mapping)

    def _on_load():
        updates = []
        for comp in persistable:
            comp_id = manager.component_to_id[comp]
            if comp_id in manager.saved_settings:
                updates.append(gr.update(value=manager.saved_settings[comp_id]))
            else:
                updates.append(gr.update())
        return updates

    for comp in persistable:
        comp.change(
            fn=_autosave,
            inputs=persistable,
            outputs=[],
            show_progress=False,
        )

    demo.load(fn=_on_load, inputs=None, outputs=persistable)
