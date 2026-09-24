from __future__ import annotations

import pytest

from agentic_kit.adapters.dummy_llm import DummyLlm
from agentic_kit.adapters.openai_llm import OpenAiLlm
from agentic_kit.composition import build_llm
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
