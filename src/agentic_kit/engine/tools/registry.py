"""Holds the tools and is the only thing allowed to run one.

Everything that could go wrong with a tool call is answered here with a result
the model can read: an unknown name, arguments that do not fit, a write that
nobody confirmed, or a tool that raised. A turn is never lost to a bad tool.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace

from pydantic import ValidationError

from ...domain.memory import VALUE_LIMIT, WorkingMemory
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

    async def execute(
        self, call: ToolCall, request: TurnRequest, memory: WorkingMemory | None = None
    ) -> ToolResult:
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

        context = ToolContext(request=request, memory=memory or WorkingMemory())
        try:
            result = await tool.execute(args, context)
        except Exception:
            logger.exception("tool %s raised", call.name)
            return ToolResult.rejected(
                ToolStatus.FAILED, f"The {call.name} tool failed. Do not assume it worked."
            )
        return self._narrow(tool, result)

    def _narrow(self, tool: Tool, result: ToolResult) -> ToolResult:
        """Hold a tool to what it declared it may remember.

        A key the tool never declared, or a value too long to be an identifier,
        is the tool misbehaving. That is the same class of problem as bad
        arguments or a raised exception, all of which are answered here.
        """
        kept = {
            key: value
            for key, value in result.learned.items()
            if key in tool.remembers and len(value) <= VALUE_LIMIT
        }
        if len(kept) == len(result.learned):
            return result
        logger.warning(
            "tool %s reported facts it may not write: %s",
            tool.name,
            sorted(set(result.learned) - set(kept)),
        )
        return replace(result, learned=kept)

    def _define(self, tool: Tool) -> ToolDefinition:
        """Fail at startup when a write tool has no way to be confirmed."""
        if tool.safety is Safety.WRITE and not issubclass(tool.args_model, ConfirmableArgs):
            raise TypeError(f"write tool {tool.name} must take ConfirmableArgs")
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.args_model.model_json_schema(),
            safety=tool.safety,
            remembers=tuple(tool.remembers),
        )


def _explain(error: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in issue['loc']) or 'arguments'}: {issue['msg']}"
        for issue in error.errors()
    )
    return f"Those arguments do not fit this tool ({problems})."
