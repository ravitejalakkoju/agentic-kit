"""Holds the tools and is the only thing allowed to run one.

Everything that could go wrong with a tool call is answered here with a result
the model can read: an unknown name, arguments that do not fit, a write that
nobody confirmed, or a tool that raised. A turn is never lost to a bad tool.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import ValidationError

from ...domain.models import TurnRequest
from ...domain.tools import Safety, ToolCall, ToolDefinition, ToolResult, ToolStatus
from ...ports.tools import ConfirmableArgs, Tool, ToolContext

logger = logging.getLogger("agentic_kit.tools")

CONFIRM_FIRST = (
    "This action changes data. Tell the customer exactly what you are about to do, "
    "get a clear yes, then call this tool again with confirmed set to true."
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._definitions = {name: self._define(tool) for name, tool in self._tools.items()}

    def catalog(self) -> tuple[ToolDefinition, ...]:
        """Every tool there is, for documentation rather than for a turn."""
        return tuple(self._definitions.values())

    def definitions(self, request: TurnRequest) -> tuple[ToolDefinition, ...]:
        """The tools this turn may use, which is what the model gets offered."""
        return tuple(
            self._definitions[name]
            for name, tool in self._tools.items()
            if tool.is_available(request)
        )

    async def execute(self, call: ToolCall, request: TurnRequest) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None or not tool.is_available(request):
            return ToolResult.rejected(
                ToolStatus.INVALID_INPUT, f"There is no tool called {call.name!r} in this turn."
            )

        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as error:
            return ToolResult.rejected(ToolStatus.INVALID_INPUT, _explain(error))

        if tool.safety is Safety.WRITE and not getattr(args, "confirmed", False):
            return ToolResult.rejected(ToolStatus.NEEDS_CONFIRMATION, CONFIRM_FIRST)

        try:
            return await tool.execute(args, ToolContext(request=request))
        except Exception:
            logger.exception("tool %s raised", call.name)
            return ToolResult.rejected(
                ToolStatus.FAILED, f"The {call.name} tool failed. Do not assume it worked."
            )

    def _define(self, tool: Tool) -> ToolDefinition:
        """Fail at startup when a write tool has no way to be confirmed."""
        if tool.safety is Safety.WRITE and not issubclass(tool.args_model, ConfirmableArgs):
            raise TypeError(f"write tool {tool.name} must take ConfirmableArgs")
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.args_model.model_json_schema(),
            safety=tool.safety,
        )


def _explain(error: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in issue['loc']) or 'arguments'}: {issue['msg']}"
        for issue in error.errors()
    )
    return f"Those arguments do not fit this tool ({problems})."
