"""The model seam. Two implementations today: deterministic and OpenAI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..domain.models import Message
from ..domain.tools import ToolCall, ToolDefinition


@dataclass(frozen=True, slots=True)
class ToolExchange:
    """One round of the tool loop: what the model asked for, and what it got.

    These stay out of the conversation history so tool traffic never crowds out
    the turns a person would want to read.
    """

    calls: tuple[ToolCall, ...]
    results: tuple[tuple[str, str], ...]
    """Call id paired with the rendered result."""


@dataclass(frozen=True, slots=True)
class LlmRequest:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    exchanges: tuple[ToolExchange, ...] = ()


@dataclass(frozen=True, slots=True)
class LlmReply:
    """Either an answer or a request to use tools, never usefully both."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LlmPort(Protocol):
    name: str

    async def complete(self, request: LlmRequest) -> LlmReply:
        """Return the model's reply, or raise ProviderError."""
        ...
