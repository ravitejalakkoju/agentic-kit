"""Persistence seams. In-memory today, SQL later, same contracts."""

from __future__ import annotations

from typing import Protocol

from ..domain.models import ConversationState, RunRecord, SopDefinition


class ConversationStore(Protocol):
    async def get(self, conversation_id: str) -> ConversationState | None: ...

    async def save(self, conversation: ConversationState) -> None: ...


class RunStore(Protocol):
    async def append(self, run: RunRecord) -> None: ...


class SopCatalog(Protocol):
    async def get(self, sop_id: str) -> SopDefinition | None: ...

    async def match(self, text: str) -> SopDefinition | None:
        """Pick the procedure for an utterance. Real matching arrives with multiple SOPs."""
        ...

    async def list_all(self) -> list[SopDefinition]: ...
