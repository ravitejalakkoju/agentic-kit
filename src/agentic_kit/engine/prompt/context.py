"""What the prompt knows about the customer beyond the message itself.

Collectors each look one thing up and return a partial bag; the pipeline merges
them. A new source of context is a new collector, not an edit to the pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from ...domain.crm import Contact
from ...domain.memory import WorkingMemory
from ...domain.models import TurnContext, TurnRequest
from ...ports.readers import ContactReader, RecordReader

logger = logging.getLogger("agentic_kit.prompt")


@dataclass(frozen=True, slots=True)
class Resource:
    """A record the turn is about, labelled with what kind of record it is."""

    kind: str
    record: BaseModel


@dataclass(frozen=True, slots=True)
class ContextBag:
    contact: Contact | None = None
    resources: tuple[Resource, ...] = ()

    def merge(self, other: ContextBag) -> ContextBag:
        return ContextBag(
            contact=other.contact or self.contact,
            resources=(*self.resources, *other.resources),
        )


class Collector(Protocol):
    name: str

    async def collect(self, request: TurnRequest, memory: WorkingMemory) -> ContextBag: ...


class ContextPipeline:
    """Runs every collector and merges what they found.

    A failing collector is logged and skipped: missing CRM data makes a worse
    reply, while failing the whole turn makes no reply at all.
    """

    def __init__(self, collectors: Sequence[Collector]) -> None:
        self._collectors = tuple(collectors)

    async def collect(self, request: TurnRequest, memory: WorkingMemory) -> ContextBag:
        bag = ContextBag()
        for collector in self._collectors:
            try:
                bag = bag.merge(await collector.collect(request, memory))
            except Exception:
                logger.warning("collector %s failed; continuing without it", collector.name)
        return bag


class ContactCollector:
    name = "contact"

    def __init__(self, reader: ContactReader) -> None:
        self._reader = reader

    async def collect(self, request: TurnRequest, memory: WorkingMemory) -> ContextBag:
        context = request.context
        identities = [
            value
            for value in (
                request.customer_id,
                context.email or memory.recall("customer_email"),
                context.phone or memory.recall("customer_phone"),
            )
            if value
        ]
        return ContextBag(contact=await self._reader.find(identities))


class ResourceCollector:
    """Fetches the record a turn refers to, by the id on the turn or the one remembered.

    The record is read again every turn. That is how the agent sees a current
    status without ever having stored one.
    """

    def __init__(
        self,
        kind: str,
        reader: RecordReader[BaseModel],
        id_of: Callable[[TurnContext], str | None],
        fact_key: str,
    ) -> None:
        self.name = kind
        self._kind = kind
        self._reader = reader
        self._id_of = id_of
        self._fact_key = fact_key

    async def collect(self, request: TurnRequest, memory: WorkingMemory) -> ContextBag:
        record_id = self._id_of(request.context) or memory.recall(self._fact_key)
        record = await self._reader.get(record_id) if record_id else None
        return ContextBag(resources=(Resource(self._kind, record),) if record else ())
