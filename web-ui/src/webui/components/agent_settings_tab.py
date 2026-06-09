import json
import os

import gradio as gr
from gradio.components import Component
from typing import Any, Dict, Optional
from src.webui.setting_utils import init_value as sv
from src.webui.webui_manager import WebuiManager
from src.utils import config
import logging
from functools import partial

logger = logging.getLogger(__name__)


def update_model_dropdown(llm_provider):
    """
    Update the model name dropdown with predefined models for the selected provider.
    """
    # Use predefined models for the selected provider
    if llm_provider in config.model_names:
        return gr.Dropdown(choices=config.model_names[llm_provider], value=config.model_names[llm_provider][0],
                           interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)


async def update_mcp_server(mcp_file: str, webui_manager: WebuiManager):
    """
    Update the MCP server.
    """
    if hasattr(webui_manager, "bu_controller") and webui_manager.bu_controller:
        logger.warning("⚠️ Close controller because mcp file has changed!")
        await webui_manager.bu_controller.close_mcp_client()
        webui_manager.bu_controller = None

    if not mcp_file or not os.path.exists(mcp_file) or not mcp_file.endswith('.json'):
        logger.warning(f"{mcp_file} is not a valid MCP file.")
        return None, gr.update(visible=False)

    with open(mcp_file, 'r') as f:
        mcp_server = json.load(f)

    return json.dumps(mcp_server, indent=2), gr.update(visible=True)


def create_agent_settings_tab(webui_manager: WebuiManager):
    """
    Creates an agent settings tab.
    """
    input_components = set(webui_manager.get_components())
    tab_components = {}

    with gr.Group():
        with gr.Column():
            goal_driven_autonomy = gr.Checkbox(
                label="Goal-driven autonomy",
                value=sv(webui_manager, "agent_settings.goal_driven_autonomy", True),
                info="Reason from live UI state (modals, scroll, multi-step flows). Appends smart defaults to system/planner prompts.",
                interactive=True,
            )
            job_application_mode = gr.Checkbox(
                label="Multi-step form mode",
                value=sv(
                    webui_manager,
                    "agent_settings.job_application_mode",
                    os.getenv("JOB_APPLICATION_MODE", "true").lower() in ("1", "true", "yes", "on"),
                ),
                info="Extra per-step briefing for wizards and applications: form progress, validation, OCR, footer controls.",
                interactive=True,
            )
            remember_session_context = gr.Checkbox(
                label="Remember session context",
                value=sv(
                    webui_manager,
                    "agent_settings.remember_session_context",
                    os.getenv("JOB_SESSION_CONTEXT", "true").lower() in ("1", "true", "yes", "on"),
                ),
                info="Persist applied/skipped jobs and agent memory across steps and task runs (saved to tmp/webui_session/).",
                interactive=True,
            )
            learn_from_user_overrides = gr.Checkbox(
                label="Learn from user overrides",
                value=sv(
                    webui_manager,
                    "agent_settings.learn_from_user_overrides",
                    os.getenv("USER_LEARNING_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
                ),
                info="Capture manual clicks, scrolls, and form edits; synthesize cross-run preference rules to tmp/webui_session/user_preferences.json.",
                interactive=True,
            )
            override_system_prompt = gr.Textbox(
                label="Override system prompt",
                lines=4,
                value=sv(webui_manager, "agent_settings.override_system_prompt", ""),
                interactive=True,
            )
            extend_system_prompt = gr.Textbox(
                label="Extend system prompt",
                lines=6,
                interactive=True,
                placeholder="Optional. Leave empty for generic web hints (forms, modals, scroll). Task-specific rules go in prompt.txt.",
                value=sv(
                    webui_manager,
                    "agent_settings.extend_system_prompt",
                    os.getenv("EXTEND_SYSTEM_PROMPT", "").strip(),
                ),
            )

    with gr.Group():
        mcp_json_file = gr.File(label="MCP server json", interactive=True, file_types=[".json"])
        mcp_server_config = gr.Textbox(
            label="MCP server",
            lines=6,
            value=sv(webui_manager, "agent_settings.mcp_server_config", ""),
            interactive=True,
            visible=False,
        )

    with gr.Group():
        with gr.Row():
            llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="LLM Provider",
                value=sv(webui_manager, "agent_settings.llm_provider", os.getenv("DEFAULT_LLM", "openai")),
                info="Select LLM provider for LLM",
                interactive=True
            )
            llm_model_name = gr.Dropdown(
                label="LLM Model Name",
                choices=config.model_names[sv(webui_manager, "agent_settings.llm_provider", os.getenv("DEFAULT_LLM", "openai"))],
                value=sv(
                    webui_manager,
                    "agent_settings.llm_model_name",
                    config.model_names[os.getenv("DEFAULT_LLM", "openai")][0],
                ),
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=sv(webui_manager, "agent_settings.llm_temperature", 0.6),
                step=0.1,
                label="LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            use_vision = gr.Checkbox(
                label="Use Vision",
                value=sv(webui_manager, "agent_settings.use_vision", True),
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            enable_ocr = gr.Checkbox(
                label="Enable OCR (pytesseract)",
                value=sv(
                    webui_manager,
                    "agent_settings.enable_ocr",
                    os.getenv("OCR_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
                ),
                info="Read visible text from screenshots each step (forms, buttons, search). Requires Tesseract installed.",
                interactive=True,
            )

            enable_html_inspect = gr.Checkbox(
                label="Enable HTML / DOM inspect",
                value=sv(
                    webui_manager,
                    "agent_settings.enable_html_inspect",
                    os.getenv("HTML_INSPECT_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
                ),
                info="Parse full page structure each step: forms, labels, values, dialogs, buttons (Playwright DOM).",
                interactive=True,
            )

            ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=sv(webui_manager, "agent_settings.ollama_num_ctx", 16000),
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            llm_base_url = gr.Textbox(
                label="Base URL",
                value=sv(webui_manager, "agent_settings.llm_base_url", ""),
                info="API endpoint URL (if required)"
            )
            llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value=sv(webui_manager, "agent_settings.llm_api_key", ""),
                info="Your API key (leave blank to use .env)"
            )

    with gr.Group():
        with gr.Row():
            planner_llm_provider = gr.Dropdown(
                choices=[provider for provider, model in config.model_names.items()],
                label="Planner LLM Provider",
                info="Select LLM provider for LLM",
                value=sv(webui_manager, "agent_settings.planner_llm_provider", None),
                interactive=True
            )
            planner_llm_model_name = gr.Dropdown(
                label="Planner LLM Model Name",
                value=sv(webui_manager, "agent_settings.planner_llm_model_name", None),
                interactive=True,
                allow_custom_value=True,
                info="Select a model in the dropdown options or directly type a custom model name"
            )
        with gr.Row():
            planner_llm_temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=sv(webui_manager, "agent_settings.planner_llm_temperature", 0.6),
                step=0.1,
                label="Planner LLM Temperature",
                info="Controls randomness in model outputs",
                interactive=True
            )

            planner_use_vision = gr.Checkbox(
                label="Use Vision(Planner LLM)",
                value=sv(webui_manager, "agent_settings.planner_use_vision", False),
                info="Enable Vision(Input highlighted screenshot into LLM)",
                interactive=True
            )

            planner_ollama_num_ctx = gr.Slider(
                minimum=2 ** 8,
                maximum=2 ** 16,
                value=sv(webui_manager, "agent_settings.planner_ollama_num_ctx", 16000),
                step=1,
                label="Ollama Context Length",
                info="Controls max context length model needs to handle (less = faster)",
                visible=False,
                interactive=True
            )

        with gr.Row():
            planner_llm_base_url = gr.Textbox(
                label="Base URL",
                value=sv(webui_manager, "agent_settings.planner_llm_base_url", ""),
                info="API endpoint URL (if required)"
            )
            planner_llm_api_key = gr.Textbox(
                label="API Key",
                type="password",
                value=sv(webui_manager, "agent_settings.planner_llm_api_key", ""),
                info="Your API key (leave blank to use .env)"
            )

    with gr.Row():
        planner_interval = gr.Slider(
            minimum=1,
            maximum=20,
            value=sv(webui_manager, "agent_settings.planner_interval", 3),
            step=1,
            label="Planner interval (steps)",
            info="Run the planning LLM every N steps when goal-driven autonomy or a planner model is set",
            interactive=True,
        )
        use_main_llm_as_planner = gr.Checkbox(
            label="Use main LLM as planner",
            value=sv(webui_manager, "agent_settings.use_main_llm_as_planner", True),
            info="When no separate planner provider is selected, reuse the main model for high-level planning",
            interactive=True,
        )

    with gr.Row():
        max_steps = gr.Slider(
            minimum=1,
            maximum=1000,
            value=sv(webui_manager, "agent_settings.max_steps", 100),
            step=1,
            label="Max Run Steps",
            info="Maximum number of steps the agent will take",
            interactive=True
        )
        max_actions = gr.Slider(
            minimum=1,
            maximum=100,
            value=sv(webui_manager, "agent_settings.max_actions", 10),
            step=1,
            label="Max Number of Actions",
            info="Maximum number of actions the agent will take per step",
            interactive=True
        )

    with gr.Row():
        max_input_tokens = gr.Number(
            label="Max Input Tokens",
            value=sv(webui_manager, "agent_settings.max_input_tokens", 128000),
            precision=0,
            interactive=True
        )
        tool_calling_method = gr.Dropdown(
            label="Tool Calling Method",
            value=sv(webui_manager, "agent_settings.tool_calling_method", "auto"),
            interactive=True,
            allow_custom_value=True,
            choices=['function_calling', 'json_mode', 'raw', 'auto', 'tools', "None"],
            visible=True
        )
    tab_components.update(dict(
        goal_driven_autonomy=goal_driven_autonomy,
        job_application_mode=job_application_mode,
        remember_session_context=remember_session_context,
        learn_from_user_overrides=learn_from_user_overrides,
        override_system_prompt=override_system_prompt,
        extend_system_prompt=extend_system_prompt,
        planner_interval=planner_interval,
        use_main_llm_as_planner=use_main_llm_as_planner,
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_temperature=llm_temperature,
        use_vision=use_vision,
        enable_ocr=enable_ocr,
        enable_html_inspect=enable_html_inspect,
        ollama_num_ctx=ollama_num_ctx,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        planner_llm_provider=planner_llm_provider,
        planner_llm_model_name=planner_llm_model_name,
        planner_llm_temperature=planner_llm_temperature,
        planner_use_vision=planner_use_vision,
        planner_ollama_num_ctx=planner_ollama_num_ctx,
        planner_llm_base_url=planner_llm_base_url,
        planner_llm_api_key=planner_llm_api_key,
        max_steps=max_steps,
        max_actions=max_actions,
        max_input_tokens=max_input_tokens,
        tool_calling_method=tool_calling_method,
        mcp_json_file=mcp_json_file,
        mcp_server_config=mcp_server_config,
    ))
    webui_manager.add_components("agent_settings", tab_components)

    llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=llm_provider,
        outputs=ollama_num_ctx
    )
    llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[llm_provider],
        outputs=[llm_model_name]
    )
    planner_llm_provider.change(
        fn=lambda x: gr.update(visible=x == "ollama"),
        inputs=[planner_llm_provider],
        outputs=[planner_ollama_num_ctx]
    )
    planner_llm_provider.change(
        lambda provider: update_model_dropdown(provider),
        inputs=[planner_llm_provider],
        outputs=[planner_llm_model_name]
    )

    async def update_wrapper(mcp_file):
        """Wrapper for handle_pause_resume."""
        update_dict = await update_mcp_server(mcp_file, webui_manager)
        yield update_dict

    mcp_json_file.change(
        update_wrapper,
        inputs=[mcp_json_file],
        outputs=[mcp_server_config, mcp_server_config]
    )
