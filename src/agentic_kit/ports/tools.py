"""The tool seam.

A tool declares its arguments as a pydantic model, so the schema the model is
shown and the validation the arguments go through are the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from ..domain.models import TurnRequest
from ..domain.tools import Safety, ToolResult


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool knows besides its arguments.

    The turn carries ids the customer never repeats, so a tool can fall back to
    the order or ticket already in context when the model leaves it out.
    """

    request: TurnRequest


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

    def is_available(self, request: TurnRequest) -> bool:
        """Whether this turn may use the tool at all."""
        ...

    async def execute(self, args: ArgsT, context: ToolContext) -> ToolResult: ...
