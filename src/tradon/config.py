"""Application configuration: secrets/env via pydantic-settings, app config via YAML."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Settings(BaseSettings):
    """Secrets and environment-specific values. Loaded from .env / process env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Default: a single SQLite file under the project (no server/Docker/VPS
    # needed). Point at Postgres instead (see docker-compose.yml) if you
    # outgrow single-machine use — nothing else in the code has to change.
    database_url: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'data' / 'tradon.db'}"

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""  # get it from the bot's /start reply, then set this and restart

    # Signs Web dashboard session cookies. Change it in .env for anything
    # beyond local personal use — the default is fine for localhost only.
    web_secret_key: str = "insecure-default-change-me-in-.env"

    # 127.0.0.1 for local dev (nothing outside the machine can reach it).
    # Docker sets WEB_HOST=0.0.0.0 (see docker-compose.yml) — binding to
    # loopback INSIDE a container makes it unreachable from outside even
    # with a port published, since the published port forwards to the
    # container's external interface, not its loopback.
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_app_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load non-secret app config (instruments, strategies) from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
