"""The vocabulary of a guardrail check: where it runs, what it saw, what follows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.models import TurnRequest


class Checkpoint(StrEnum):
    """Where in a turn a detector runs, which decides what text it is handed."""

    INPUT = "input"
    PROMPT = "prompt"
    OUTPUT = "output"
    MEMORY = "memory"
    """A fact on its way into storage, where it would join every later prompt."""
    KNOWLEDGE = "knowledge"
    """A document or a passage, which is third-party text the customer never wrote."""


class Level(StrEnum):
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


class Action(StrEnum):
    """What a failure costs the customer: a refusal, or a human."""

    BLOCK = "block"
    HANDOFF = "handoff"


class Mode(StrEnum):
    ENFORCE = "enforce"
    OBSERVE = "observe"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class Finding:
    detector_id: str
    checkpoint: Checkpoint
    level: Level
    message: str
    action: Action = Action.BLOCK
    """Only read when the level is FAIL and the checkpoint is enforced."""


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    findings: tuple[Finding, ...] = ()

    @property
    def failure(self) -> Finding | None:
        """The finding that stopped the turn, if one did."""
        return next((f for f in self.findings if f.level is Level.FAIL), None)


@dataclass(frozen=True, slots=True)
class GuardrailContext:
    """One detector's view of whatever is being checked.

    `text` is what the checkpoint is about: the customer's message, the
    assembled prompt, the drafted reply, or a document on its way into the
    knowledge base. That last one has no turn behind it, which is the whole
    reason the request is optional.
    """

    checkpoint: Checkpoint
    text: str
    request: TurnRequest | None = None
