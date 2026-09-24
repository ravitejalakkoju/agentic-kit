"""A model that needs no key, no network, and returns the same thing every time."""

from __future__ import annotations

from collections.abc import Sequence

from ..domain.models import Message, Role


class DummyLlm:
    name = "dummy"

    async def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        last_user = next(
            (message.text for message in reversed(messages) if message.role is Role.USER),
            "",
        )
        return (
            f'You said: "{last_user}". '
            "This is the offline model; set a real OPENAI_API_KEY for live replies."
        )
