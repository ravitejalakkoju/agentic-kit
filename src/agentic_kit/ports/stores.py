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
    """Holds the procedures. Deliberately cannot pick between them.

    Choosing one for an utterance is a judgement over what was said, which is
    the matcher's job; a store that also ranked would be two things.
    """

    async def get(self, sop_id: str) -> SopDefinition | None: ...

    async def list_all(self) -> list[SopDefinition]: ...
