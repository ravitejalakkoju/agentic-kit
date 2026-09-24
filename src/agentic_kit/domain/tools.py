"""What a tool is, and what came back from running one."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any


class Safety(StrEnum):
    """How much a tool can cost if the model gets it wrong."""

    READ = "read"
    WRITE = "write"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    NEEDS_CONFIRMATION = "needs_confirmation"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A tool as the model sees it. `parameters` is a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    safety: Safety
    remembers: tuple[str, ...] = ()
    """The fact keys this tool may write, so what it can put in a later prompt is readable."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    status: ToolStatus
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    learned: Mapping[str, str] = field(default_factory=dict)
    """Identifiers this call established. Read by memory, never shown to the model."""
    needs: tuple[str, ...] = ()
    """Keys the call could not proceed without, so the conversation can wait on them."""

    @classmethod
    def ok(cls, message: str, **data: Any) -> ToolResult:
        return cls(status=ToolStatus.SUCCESS, message=message, data=data)

    @classmethod
    def missing(cls, message: str) -> ToolResult:
        return cls(status=ToolStatus.NOT_FOUND, message=message)

    @classmethod
    def rejected(cls, status: ToolStatus, message: str) -> ToolResult:
        return cls(status=status, message=message)

    @classmethod
    def needs_input(cls, key: str, message: str) -> ToolResult:
        """Stop for something only the customer can supply.

        The message is both what the model reads now and what the conversation
        is recorded as waiting on, so the two cannot disagree.
        """
        return cls(status=ToolStatus.INVALID_INPUT, message=message, needs=(key,))

    def remembering(self, **facts: str | None) -> ToolResult:
        """The same result, plus what it taught the conversation.

        A key with nothing behind it is not a fact, so it is dropped rather
        than stored as an empty one.
        """
        found = {key: value for key, value in facts.items() if value}
        return replace(self, learned={**self.learned, **found}) if found else self

    @property
    def succeeded(self) -> bool:
        return self.status is ToolStatus.SUCCESS

    def for_model(self) -> str:
        """The result as the model reads it: always JSON, always with a status."""
        body: dict[str, Any] = {"status": self.status, "message": self.message}
        if self.data:
            body |= self.data
        return json.dumps(body, ensure_ascii=False, default=str)
