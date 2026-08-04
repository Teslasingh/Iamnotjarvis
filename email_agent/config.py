from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CLIENT_SECRET_FILE = next(BASE_DIR.glob("client_secret_*.json"), BASE_DIR / "client_secret.json")
TOKEN_FILE = DATA_DIR / "token.json"
OAUTH_STATE_FILE = DATA_DIR / "oauth_state.json"
DB_FILE = DATA_DIR / "email_agent.sqlite3"
PROFILE_FILE = DATA_DIR / "profile.json"

load_dotenv(ENV_FILE)


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class Settings:
    app_name = "Gmail Job Assistant"
    host = os.getenv("EMAIL_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("EMAIL_AGENT_PORT", "8000"))
    base_url = os.getenv("EMAIL_AGENT_BASE_URL", f"http://{host}:{port}")
    oauth_redirect_uri = os.getenv("GMAIL_REDIRECT_URI", f"{base_url}/oauth2callback")
    openai_api_key = _first_env("OPENAI_API_KEY", "OPENAI_KEY", "OPENAI_API_TOKEN", "LLM_API_KEY")
    openai_base_url = _first_env("OPENAI_BASE_URL", "OPENAI_API_BASE")
    openai_model = _first_env("OPENAI_MODEL", "LLM_MODEL", default="gpt-4o-mini")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_version = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
    azure_openai_deployment = os.getenv("DEPLOYMENT_NAME", "")
    default_sync_query = os.getenv("GMAIL_SYNC_QUERY", "in:inbox newer_than:5d")
    default_sync_limit = int(os.getenv("GMAIL_SYNC_LIMIT", "25"))

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key or (self.azure_openai_api_key and self.azure_openai_endpoint))

    @property
    def llm_model(self) -> str:
        return self.azure_openai_deployment or self.openai_model

    @property
    def llm_supports_custom_temperature(self) -> bool:
        return "gpt-5" not in self.llm_model.lower()

    def chat_temperature(self, desired: float) -> dict[str, float]:
        if self.llm_supports_custom_temperature:
            return {"temperature": desired}
        return {}


settings = Settings()


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
