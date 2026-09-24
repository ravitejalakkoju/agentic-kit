"""The instructions a turn is prompted with, resolved once so sections only render."""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.models import SopDefinition, TurnRequest

DEFAULT_RESPONSE_STRATEGY = "Resolve the customer's latest message in as few turns as possible."


@dataclass(frozen=True, slots=True)
class PromptPolicy:
    sop: SopDefinition
    channel: str
    guardrail_rules: tuple[str, ...]
    response_strategy: str

    @classmethod
    def for_turn(
        cls, sop: SopDefinition, request: TurnRequest, rules: tuple[str, ...] = ()
    ) -> PromptPolicy:
        return cls(
            sop=sop,
            channel=request.context.channel,
            guardrail_rules=rules,
            response_strategy=sop.response_strategy or DEFAULT_RESPONSE_STRATEGY,
        )
