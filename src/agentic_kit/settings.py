"""Process configuration read from the environment."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

DUMMY_KEY_PREFIX = "dummy"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 3200
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str | None = None
    openai_timeout_seconds: float = 30.0
    sqlite_path: str = ""

    @property
    def has_live_llm(self) -> bool:
        """A real key is present, so the OpenAI adapter can be used."""
        key = self.openai_api_key.strip()
        return bool(key) and not key.startswith(DUMMY_KEY_PREFIX)


def load_settings() -> Settings:
    return Settings()
