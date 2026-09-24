"""The shape of what we send OpenAI, which no other test would catch breaking."""

from __future__ import annotations

import json

import pytest

from agentic_kit.adapters.openai_llm import OpenAiLlm, _as_call, _as_function
from agentic_kit.domain.models import Message, Role
from agentic_kit.domain.tools import Safety, ToolCall, ToolDefinition
from agentic_kit.errors import ProviderError
from agentic_kit.ports.llm import LlmRequest, ToolExchange


class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeRawCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = FakeFunction(name, arguments)


@pytest.fixture
def llm() -> OpenAiLlm:
    return OpenAiLlm(api_key="sk-not-used", model="gpt-4.1-mini")


def test_history_follows_the_system_prompt(llm: OpenAiLlm) -> None:
    request = LlmRequest(
        system="SYSTEM",
        messages=(
            Message(role=Role.USER, text="hi"),
            Message(role=Role.AGENT, text="hello"),
        ),
    )

    assert llm._messages(request) == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_a_finished_round_replays_as_an_ask_then_an_answer(llm: OpenAiLlm) -> None:
    exchange = ToolExchange(
        calls=(ToolCall(id="c1", name="track_order", arguments={"order_id": "1001"}),),
        results=(("c1", '{"status": "success"}'),),
    )
    request = LlmRequest(system="S", messages=(), exchanges=(exchange,))

    _, ask, answer = llm._messages(request)

    assert ask["role"] == "assistant"
    assert ask["content"] is None
    assert json.loads(ask["tool_calls"][0]["function"]["arguments"]) == {"order_id": "1001"}
    assert answer == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": '{"status": "success"}',
    }


def test_a_tool_becomes_a_function_definition() -> None:
    definition = ToolDefinition(
        name="track_order",
        description="Track an order.",
        parameters={"type": "object", "properties": {}},
        safety=Safety.READ,
    )

    assert _as_function(definition) == {
        "type": "function",
        "function": {
            "name": "track_order",
            "description": "Track an order.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_a_returned_call_is_parsed() -> None:
    call = _as_call(FakeRawCall("c1", "track_order", '{"order_id": "1001"}'))

    assert call == ToolCall(id="c1", name="track_order", arguments={"order_id": "1001"})


def test_a_call_with_no_arguments_is_still_readable() -> None:
    assert _as_call(FakeRawCall("c1", "lookup_contact", "")).arguments == {}


def test_unreadable_arguments_are_a_provider_error() -> None:
    with pytest.raises(ProviderError, match="unreadable arguments"):
        _as_call(FakeRawCall("c1", "track_order", "{not json"))
