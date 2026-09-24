"""The registry is the only thing that runs a tool, so it owns every refusal."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, Field

from agentic_kit.domain.models import TurnRequest
from agentic_kit.domain.tools import Safety, ToolCall, ToolResult, ToolStatus
from agentic_kit.engine.tools.registry import ToolRegistry
from agentic_kit.ports.tools import ConfirmableArgs, ToolContext

REQUEST = TurnRequest(conversation_id="conv-1", customer_id="cust-1", text="hi")


class EchoArgs(BaseModel):
    value: str = Field(description="Anything at all.")


class EchoTool:
    name = "echo"
    description = "Repeat a value back."
    safety = Safety.READ
    args_model = EchoArgs

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.ran_with: list[str] = []

    def is_available(self, request: TurnRequest) -> bool:
        return self._available

    async def execute(self, args: EchoArgs, context: ToolContext) -> ToolResult:
        self.ran_with.append(args.value)
        return ToolResult.ok("echoed", value=args.value)


class DeleteArgs(ConfirmableArgs):
    target: str


class DeleteTool:
    name = "delete"
    description = "Delete something."
    safety = Safety.WRITE
    args_model = DeleteArgs

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: DeleteArgs, context: ToolContext) -> ToolResult:
        self.deleted.append(args.target)
        return ToolResult.ok("deleted")


class ExplodingTool:
    name = "explode"
    description = "Always raises."
    safety = Safety.READ
    args_model = EchoArgs

    def is_available(self, request: TurnRequest) -> bool:
        return True

    async def execute(self, args: EchoArgs, context: ToolContext) -> ToolResult:
        raise RuntimeError("the vendor is down")


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id="call-1", name=name, arguments=arguments)


def test_schema_comes_from_the_args_model() -> None:
    [definition] = ToolRegistry([EchoTool()]).catalog()

    assert definition.parameters["properties"]["value"]["description"] == "Anything at all."
    assert definition.parameters["required"] == ["value"]


def test_catalog_lists_every_tool_but_definitions_only_the_available_ones() -> None:
    registry = ToolRegistry([EchoTool(available=False), DeleteTool()])

    assert {d.name for d in registry.catalog()} == {"echo", "delete"}
    assert {d.name for d in registry.definitions(REQUEST)} == {"delete"}


async def test_a_good_call_runs_the_tool() -> None:
    tool = EchoTool()

    result = await ToolRegistry([tool]).execute(call("echo", value="hello"), REQUEST)

    assert result.succeeded
    assert tool.ran_with == ["hello"]


async def test_an_unknown_tool_is_refused_not_raised() -> None:
    result = await ToolRegistry([EchoTool()]).execute(call("nope"), REQUEST)

    assert result.status is ToolStatus.INVALID_INPUT
    assert "nope" in result.message


async def test_an_unavailable_tool_cannot_be_called() -> None:
    tool = EchoTool(available=False)

    result = await ToolRegistry([tool]).execute(call("echo", value="hi"), REQUEST)

    assert result.status is ToolStatus.INVALID_INPUT
    assert tool.ran_with == []


async def test_bad_arguments_become_a_readable_refusal() -> None:
    tool = EchoTool()

    result = await ToolRegistry([tool]).execute(call("echo"), REQUEST)

    assert result.status is ToolStatus.INVALID_INPUT
    assert "value" in result.message
    assert tool.ran_with == []


async def test_a_raising_tool_becomes_a_failure(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="agentic_kit.tools"):
        result = await ToolRegistry([ExplodingTool()]).execute(call("explode", value="x"), REQUEST)

    assert result.status is ToolStatus.FAILED
    assert "the vendor is down" in caplog.text


async def test_a_write_tool_will_not_run_unconfirmed() -> None:
    tool = DeleteTool()

    result = await ToolRegistry([tool]).execute(call("delete", target="order-1"), REQUEST)

    assert result.status is ToolStatus.NEEDS_CONFIRMATION
    assert tool.deleted == []


async def test_a_confirmed_write_tool_runs() -> None:
    tool = DeleteTool()

    result = await ToolRegistry([tool]).execute(
        call("delete", target="order-1", confirmed=True), REQUEST
    )

    assert result.succeeded
    assert tool.deleted == ["order-1"]


def test_a_write_tool_without_a_confirmation_field_is_rejected_at_startup() -> None:
    class Unconfirmable:
        name = "risky"
        description = "No way to confirm."
        safety = Safety.WRITE
        args_model = EchoArgs

        def is_available(self, request: TurnRequest) -> bool:
            return True

        async def execute(self, args: EchoArgs, context: ToolContext) -> ToolResult:
            return ToolResult.ok("done")

    with pytest.raises(TypeError, match="must take ConfirmableArgs"):
        ToolRegistry([Unconfirmable()])


def test_an_empty_registry_offers_nothing() -> None:
    assert ToolRegistry().definitions(REQUEST) == ()


def test_results_reach_the_model_as_json_with_a_status() -> None:
    rendered = ToolResult.ok("found it", order={"id": "1001"}).for_model()

    assert '"status": "success"' in rendered
    assert '"id": "1001"' in rendered
