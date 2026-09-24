"""Test doubles for the ports, and the one helper that drives a turn."""

from __future__ import annotations

from collections import deque
from itertools import count
from typing import Any

from agentic_kit.composition import Components
from agentic_kit.domain.models import FlowKind, TurnContext, TurnRequest, TurnResult
from agentic_kit.domain.tools import ToolCall
from agentic_kit.ports.llm import LlmReply, LlmRequest

_ids = count(1)


async def send(
    components: Components,
    text: str,
    *,
    conversation_id: str = "conv-1",
    customer_id: str = "cust-1",
    context: TurnContext | None = None,
) -> TurnResult:
    """One turn in at the front door, the way the HTTP layer sends it."""
    request = TurnRequest(
        conversation_id=conversation_id,
        customer_id=customer_id,
        text=text,
        context=context or TurnContext(),
    )
    return await components.engine.handle(FlowKind.CONVERSATION, request)


def says(text: str) -> LlmReply:
    return LlmReply(text=text)


def calls(*wanted: ToolCall) -> LlmReply:
    return LlmReply(tool_calls=wanted)


def tool_call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call-{next(_ids)}", name=name, arguments=arguments)


class ScriptedLlm:
    """Plays back queued replies in order and records every request it receives.

    A queued exception is raised instead of returned, which is how tests make
    the provider fail. A queued string is shorthand for a plain reply. Running
    out of script fails the test loudly rather than inventing an answer.
    """

    name = "scripted"

    def __init__(self, *steps: str | LlmReply | Exception) -> None:
        self._steps: deque[str | LlmReply | Exception] = deque(steps)
        self.calls: list[LlmRequest] = []

    def queue(self, *steps: str | LlmReply | Exception) -> None:
        self._steps.extend(steps)

    async def complete(self, request: LlmRequest) -> LlmReply:
        self.calls.append(request)
        if not self._steps:
            raise AssertionError("ScriptedLlm was called with nothing left in its script")
        step = self._steps.popleft()
        if isinstance(step, Exception):
            raise step
        return says(step) if isinstance(step, str) else step
