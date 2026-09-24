"""A model that needs no key, no network, and returns the same thing every time."""

from __future__ import annotations

from ..domain.models import Role
from ..ports.llm import LlmReply, LlmRequest


class DummyLlm:
    name = "dummy"

    async def complete(self, request: LlmRequest) -> LlmReply:
        last_user = next(
            (message.text for message in reversed(request.messages) if message.role is Role.USER),
            "",
        )
        return LlmReply(
            text=(
                f'You said: "{last_user}". '
                "This is the offline model; set a real OPENAI_API_KEY for live replies."
            )
        )
