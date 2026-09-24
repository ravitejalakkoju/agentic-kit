"""The tool seam.

A tool declares its arguments as a pydantic model, so the schema the model is
shown and the validation the arguments go through are the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from ..domain.memory import WorkingMemory
from ..domain.models import TurnRequest
from ..domain.tools import Safety, ToolResult


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool knows besides its arguments.

    The turn carries ids the customer never repeats and memory carries the ones
    an earlier tool established, so a tool can fill in an argument the model
    left out rather than asking for it again.
    """

    request: TurnRequest
    memory: WorkingMemory = field(default_factory=WorkingMemory)

    def resolve(self, key: str, *candidates: str | None) -> str | None:
        """The first value this turn supplied, or else what an earlier tool learned.

        This turn wins on purpose: a caller who names an order means that one.
        """
        return next((value for value in candidates if value), None) or self.memory.recall(key)


class ConfirmableArgs(BaseModel):
    """Base for tools that change something.

    The registry refuses to run a write tool until this is true, so the model
    has to ask the customer before it can act.
    """

    confirmed: bool = False


class Tool[ArgsT: BaseModel](Protocol):
    name: str
    description: str
    safety: Safety
    args_model: type[ArgsT]
    remembers: tuple[str, ...]
    """The fact keys this tool may write. The registry drops anything else it reports."""

    def is_available(self, request: TurnRequest) -> bool:
        """Whether this turn may use the tool at all."""
        ...

    async def execute(self, args: ArgsT, context: ToolContext) -> ToolResult: ...
