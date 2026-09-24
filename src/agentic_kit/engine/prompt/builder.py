"""Builds the system prompt for one turn: collect context, render sections, join."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...domain.memory import WorkingMemory
from ...domain.models import SopDefinition, TurnRequest
from ...domain.tools import ToolDefinition
from .context import ContextBag, ContextPipeline
from .policy import PromptPolicy
from .sections import PromptInput, Section


@dataclass(frozen=True, slots=True)
class RenderedSection:
    key: str
    title: str
    priority: int
    content: str


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    system: str
    sections: tuple[RenderedSection, ...]
    context: ContextBag


class PromptBuilder:
    def __init__(
        self,
        context: ContextPipeline,
        sections: Sequence[Section],
        rules: tuple[str, ...] = (),
    ) -> None:
        self._context = context
        self._sections = tuple(sorted(sections, key=lambda section: section.priority))
        self._rules = rules

    async def build(
        self,
        *,
        sop: SopDefinition,
        request: TurnRequest,
        tools: tuple[ToolDefinition, ...] = (),
        memory: WorkingMemory | None = None,
    ) -> BuiltPrompt:
        memory = memory or WorkingMemory()
        prompt = PromptInput(
            request=request,
            policy=PromptPolicy.for_turn(sop, request, self._rules),
            context=await self._context.collect(request, memory),
            tools=tools,
            memory=memory,
        )
        rendered = tuple(
            RenderedSection(section.key, section.title, section.priority, content)
            for section in self._sections
            if (content := section.render(prompt))
        )
        system = "\n\n".join(f"{s.title}:\n{s.content}" for s in rendered)
        return BuiltPrompt(system=system, sections=rendered, context=prompt.context)
