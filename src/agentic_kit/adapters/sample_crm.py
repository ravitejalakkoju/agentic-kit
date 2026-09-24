"""In-memory readers over seeded records, standing in for a real CRM."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from pydantic import BaseModel

from ..domain.crm import Contact, Ticket


class InMemoryContactReader:
    def __init__(self, contacts: Iterable[Contact]) -> None:
        self._contacts = list(contacts)

    async def find(self, identities: Iterable[str]) -> Contact | None:
        wanted = set(identities)
        return next((c.model_copy() for c in self._contacts if c.identities & wanted), None)


class InMemoryRecordReader[RecordT: BaseModel]:
    def __init__(self, records: Iterable[RecordT], id_of: Callable[[RecordT], str]) -> None:
        self._records = {id_of(record): record for record in records}

    async def get(self, record_id: str) -> RecordT | None:
        record = self._records.get(record_id)
        return record.model_copy() if record else None


class InMemoryTicketNotes:
    """Notes written against seeded tickets, so a write tool has somewhere to go."""

    def __init__(self, tickets: InMemoryRecordReader[Ticket]) -> None:
        self._tickets = tickets
        self.notes: dict[str, list[str]] = {}

    async def add_note(self, ticket_id: str, note: str) -> bool:
        if await self._tickets.get(ticket_id) is None:
            return False
        self.notes.setdefault(ticket_id, []).append(note)
        return True
