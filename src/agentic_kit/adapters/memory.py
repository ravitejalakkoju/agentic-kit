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
            self.records.append(run.model_copy(deep=True))

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._lock:
            run = next((run for run in self.records if run.run_id == run_id), None)
            return run.model_copy(deep=True) if run else None

    async def list_for(self, conversation_id: str) -> list[RunRecord]:
        async with self._lock:
            return [
                run.model_copy(deep=True)
                for run in self.records
                if run.conversation_id == conversation_id
            ]


class InMemorySopCatalog:
    """Holds the seeded procedures, in the order they were seeded."""

    def __init__(self, sops: list[SopDefinition]) -> None:
        self._sops = {sop.sop_id: sop for sop in sops}

    async def get(self, sop_id: str) -> SopDefinition | None:
        return self._sops.get(sop_id)

    async def list_all(self) -> list[SopDefinition]:
        return list(self._sops.values())
