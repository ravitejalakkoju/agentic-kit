"""Test doubles for the ports."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from agentic_kit.domain.models import Message


@dataclass(frozen=True, slots=True)
class LlmCall:
    system: str
    messages: list[Message]


class ScriptedLlm:
    """Plays back queued replies in order and records every call it receives.

    A queued exception is raised instead of returned, which is how tests make
    the provider fail. Running out of script fails the test loudly rather than
    inventing a reply.
    """

    name = "scripted"

    def __init__(self, *steps: str | Exception) -> None:
        self._steps: deque[str | Exception] = deque(steps)
        self.calls: list[LlmCall] = []

    def queue(self, *steps: str | Exception) -> None:
        self._steps.extend(steps)

    async def complete(self, *, system: str, messages: Sequence[Message]) -> str:
        self.calls.append(LlmCall(system=system, messages=list(messages)))
        if not self._steps:
            raise AssertionError("ScriptedLlm was called with nothing left in its script")
        step = self._steps.popleft()
        if isinstance(step, Exception):
            raise step
        return step
