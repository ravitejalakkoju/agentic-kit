"""Builds the system prompt for one turn: collect context, render sections, join."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...domain.models import SopDefinition, TurnRequest
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
    def __init__(self, context: ContextPipeline, sections: Sequence[Section]) -> None:
        self._context = context
        self._sections = tuple(sorted(sections, key=lambda section: section.priority))

    async def build(self, *, sop: SopDefinition, request: TurnRequest) -> BuiltPrompt:
        prompt = PromptInput(
            request=request,
            policy=PromptPolicy.for_turn(sop, request),
            context=await self._context.collect(request),
        )
        rendered = tuple(
            RenderedSection(section.key, section.title, section.priority, content)
            for section in self._sections
            if (content := section.render(prompt))
        )
        system = "\n\n".join(f"{s.title}:\n{s.content}" for s in rendered)
        return BuiltPrompt(system=system, sections=rendered, context=prompt.context)
