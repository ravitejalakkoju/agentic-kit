from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from agentic_kit.composition import Components, build
from agentic_kit.main import create_app
from agentic_kit.settings import Settings

from .fakes import ScriptedLlm


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="dummy-test-key")


@pytest.fixture
def llm() -> ScriptedLlm:
    return ScriptedLlm()


@pytest.fixture
def components(settings: Settings, llm: ScriptedLlm) -> Components:
    return build(settings, llm)


@pytest.fixture
def client(settings: Settings, llm: ScriptedLlm) -> Iterator[TestClient]:
    with TestClient(create_app(settings, llm)) as test_client:
        yield test_client
