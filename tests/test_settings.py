from __future__ import annotations

from pathlib import Path

import pytest

from agentic_kit.adapters.dummy_llm import DummyLlm
from agentic_kit.adapters.memory import InMemoryConversationStore, InMemoryRunStore
from agentic_kit.adapters.openai_llm import OpenAiLlm
from agentic_kit.adapters.sqlite import SqliteConversationStore, SqliteRunStore
from agentic_kit.composition import build_llm, build_stores
from agentic_kit.settings import Settings


@pytest.mark.parametrize("key", ["", "   ", "dummy", "dummy-openai-key"])
def test_missing_or_dummy_key_uses_the_offline_model(key: str) -> None:
    settings = Settings(_env_file=None, openai_api_key=key)

    assert settings.has_live_llm is False
    assert isinstance(build_llm(settings), DummyLlm)


def test_real_looking_key_uses_openai() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test")

    assert settings.has_live_llm is True
    assert isinstance(build_llm(settings), OpenAiLlm)


def test_empty_sqlite_path_keeps_both_stores_in_memory() -> None:
    conversations, runs = build_stores(Settings(_env_file=None, sqlite_path=""))

    assert isinstance(conversations, InMemoryConversationStore)
    assert isinstance(runs, InMemoryRunStore)


def test_sqlite_path_selects_both_disk_stores(tmp_path: Path) -> None:
    conversations, runs = build_stores(
        Settings(_env_file=None, sqlite_path=str(tmp_path / "agentic-kit.db"))
    )

    assert isinstance(conversations, SqliteConversationStore)
    assert isinstance(runs, SqliteRunStore)
