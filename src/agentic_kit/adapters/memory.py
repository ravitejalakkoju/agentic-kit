"""In-memory stores.

They copy on read and write so that callers cannot mutate stored state by
accident, which is how a database-backed store will behave.
"""

from __future__ import annotations

import asyncio

from ..domain.models import ConversationState, RunRecord, SopDefinition


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._conversations: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get(self, conversation_id: str) -> ConversationState | None:
        async with self._lock:
            conversation = self._conversations.get(conversation_id)
            return conversation.model_copy(deep=True) if conversation else None

    async def save(self, conversation: ConversationState) -> None:
        async with self._lock:
            self._conversations[conversation.conversation_id] = conversation.model_copy(deep=True)


class InMemoryRunStore:
    def __init__(self) -> None:
        self.records: list[RunRecord] = []
        self._lock = asyncio.Lock()

    async def append(self, run: RunRecord) -> None:
        async with self._lock:
            self.records.append(run)


class InMemorySopCatalog:
    """Holds the seeded procedures.

    `match` returns the only procedure there is. Ranking by trigger examples
    belongs here when a second procedure exists.
    """

    def __init__(self, sops: list[SopDefinition]) -> None:
        self._sops = {sop.sop_id: sop for sop in sops}

    async def get(self, sop_id: str) -> SopDefinition | None:
        return self._sops.get(sop_id)

    async def match(self, text: str) -> SopDefinition | None:
        return next(iter(self._sops.values()), None)

    async def list_all(self) -> list[SopDefinition]:
        return list(self._sops.values())
