"""What a conversation remembers between turns.

Tools are the only writers, and they write identifiers: an order reference, a
ticket number, a phone number. Anything that changes on its own is deliberately
not remembered, because a remembered value that has gone stale is worse than no
memory at all. It reads as current, so it can talk the model out of the tool
call that would have refreshed it.

A value that replaces another does not erase it. The old one moves to
`superseded`, so a wrong answer can be traced back to what the agent believed
and to the turn that taught it.
"""

from __future__ import annotations

from collections.abc import Container, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .tools import ToolResult

FACT_LIMIT = 20
"""Active facts kept per conversation. Past this the oldest are archived."""

ARCHIVE_LIMIT = 40
"""Displaced facts kept for tracing, newest first."""

VALUE_LIMIT = 200
"""A fact is an identifier. Anything longer is malformed rather than merely wrong."""


def _now() -> datetime:
    # Not shared with models.py: that module imports this one, not the reverse.
    return datetime.now(UTC)


class Fact(BaseModel):
    """Something a tool established, and where it came from."""

    key: str
    value: str
    source: str
    """The tool that established it."""
    run_id: str | None = None
    """The turn it was learned on, stamped when the update is committed."""
    at: datetime = Field(default_factory=_now)


class PendingInput(BaseModel):
    """Something a tool could not proceed without."""

    key: str
    prompt: str
    """The tool's own words, so what the model is told and what the conversation waits on
    cannot drift apart."""
    source: str
    at: datetime = Field(default_factory=_now)


@dataclass(frozen=True, slots=True)
class MemoryUpdate:
    """What one turn's tools reported, before any of it is committed."""

    facts: tuple[Fact, ...] = ()
    pending: tuple[PendingInput, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.facts or self.pending)

    @classmethod
    def from_tools(cls, reported: Iterable[tuple[str, ToolResult]]) -> MemoryUpdate:
        """Credit what each tool reported to its own name."""
        facts: list[Fact] = []
        pending: list[PendingInput] = []
        for source, result in reported:
            facts.extend(
                Fact(key=key, value=value, source=source) for key, value in result.learned.items()
            )
            pending.extend(
                PendingInput(key=key, prompt=result.message, source=source) for key in result.needs
            )
        return cls(facts=tuple(facts), pending=tuple(pending))

    def then(self, other: MemoryUpdate) -> MemoryUpdate:
        """Both updates in order, so a later tool round lands after an earlier one."""
        return MemoryUpdate(
            facts=(*self.facts, *other.facts),
            pending=(*self.pending, *other.pending),
        )

    def without(self, keys: Container[str]) -> MemoryUpdate:
        """The same update with some facts left out, after screening rejected them."""
        return MemoryUpdate(
            facts=tuple(fact for fact in self.facts if fact.key not in keys),
            pending=self.pending,
        )


class WorkingMemory(BaseModel):
    """One active value per key, what it displaced, and what is still missing."""

    facts: dict[str, Fact] = Field(default_factory=dict)
    superseded: list[Fact] = Field(default_factory=list)
    """Values no longer active, newest first. For people reading back, never for the prompt."""
    pending: dict[str, PendingInput] = Field(default_factory=dict)

    def recall(self, key: str) -> str | None:
        """Only ever the active value."""
        fact = self.facts.get(key)
        return fact.value if fact else None

    def apply(self, update: MemoryUpdate, run_id: str | None = None) -> None:
        """Commit an update, archiving whatever it displaces.

        Facts land first, so a turn where one tool asked for a key and another
        answered it leaves nothing outstanding.
        """
        for fact in update.facts:
            self._record(fact.model_copy(update={"run_id": run_id}))
        for want in update.pending:
            if want.key not in self.facts:
                self.pending[want.key] = want
        self._trim()

    def _record(self, fact: Fact) -> None:
        """Re-learning the same value is not a change, so the original provenance stands."""
        known = self.facts.get(fact.key)
        if known is None or known.value != fact.value:
            if known is not None:
                self._archive(known)
            self.facts[fact.key] = fact
        self.pending.pop(fact.key, None)

    def _archive(self, fact: Fact) -> None:
        self.superseded = [fact, *self.superseded][:ARCHIVE_LIMIT]

    def _trim(self) -> None:
        """Evict the oldest facts, keeping them traceable in the archive."""
        excess = len(self.facts) - FACT_LIMIT
        if excess <= 0:
            return
        for fact in sorted(self.facts.values(), key=lambda fact: fact.at)[:excess]:
            del self.facts[fact.key]
            self._archive(fact)
