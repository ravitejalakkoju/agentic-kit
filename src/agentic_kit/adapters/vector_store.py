"""A vector store that is a list and a loop.

Exact search rather than approximate, which is the right trade for a few
hundred passages and the wrong one for a million. That is the point at which
this seam earns a real index behind it, and nothing above it has to notice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from ..domain.knowledge import Chunk, Passage, Shelved
from ..errors import KnowledgeError
from ..ports.knowledge import Vector


@dataclass(frozen=True, slots=True)
class _Entry:
    chunk: Chunk
    vector: Vector


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._dimensions: int | None = None
        self._lock = asyncio.Lock()

    async def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise KnowledgeError(f"{len(chunks)} passages but {len(vectors)} vectors")
        async with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._check(vector)
                self._entries.append(_Entry(chunk, tuple(vector)))

    async def search(self, vector: Vector, limit: int) -> list[Passage]:
        async with self._lock:
            entries = list(self._entries)
        scored = [Passage(entry.chunk, _similarity(vector, entry.vector)) for entry in entries]
        scored.sort(key=lambda passage: passage.score, reverse=True)
        return scored[:limit]

    async def remove(self, document_id: str) -> int:
        async with self._lock:
            kept = [entry for entry in self._entries if entry.chunk.document_id != document_id]
            dropped = len(self._entries) - len(kept)
            self._entries = kept
            return dropped

    async def shelved(self) -> list[Shelved]:
        async with self._lock:
            entries = list(self._entries)

        counts: dict[str, int] = {}
        titles: dict[str, Chunk] = {}
        for entry in entries:
            counts[entry.chunk.document_id] = counts.get(entry.chunk.document_id, 0) + 1
            titles.setdefault(entry.chunk.document_id, entry.chunk)
        return [
            Shelved(
                document_id=document_id,
                title=titles[document_id].title,
                source=titles[document_id].source,
                passages=count,
            )
            for document_id, count in counts.items()
        ]

    def _check(self, vector: Vector) -> None:
        """Vectors from two different embedders cannot be compared.

        Swapping the embedding model without clearing the store would leave
        search quietly wrong rather than loudly broken, so it is refused here.
        """
        if self._dimensions is None:
            self._dimensions = len(vector)
        elif len(vector) != self._dimensions:
            raise KnowledgeError(
                f"store holds {self._dimensions}-dimension vectors, got {len(vector)}: "
                "the embedder changed and the store was not rebuilt"
            )


def _similarity(left: Vector, right: Vector) -> float:
    """Cosine: how far apart the two point, ignoring how long either is."""
    size = _length(left) * _length(right)
    return sum(a * b for a, b in zip(left, right, strict=True)) / size if size else 0.0


def _length(vector: Vector) -> float:
    return sqrt(sum(value * value for value in vector))
