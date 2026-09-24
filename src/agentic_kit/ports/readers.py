"""Read seams onto the customer's systems. The engine depends on these, never on a CRM."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from pydantic import BaseModel

from ..domain.crm import Contact


class ContactReader(Protocol):
    async def find(self, identities: Iterable[str]) -> Contact | None:
        """The contact matching any of the given ids, emails, or phone numbers."""
        ...


class RecordReader[RecordT: BaseModel](Protocol):
    """Looks one record up by id. Orders and tickets share this shape."""

    async def get(self, record_id: str) -> RecordT | None: ...


class TicketWriter(Protocol):
    async def add_note(self, ticket_id: str, note: str) -> bool:
        """Record a note against a ticket. False when there is no such ticket."""
        ...
