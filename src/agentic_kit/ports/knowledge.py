"""Turning text into numbers, and finding the nearest ones again.

Two seams rather than one. Which model does the embedding and which database
holds the result are separate decisions, and a team changes them at separate
times.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..domain.knowledge import Chunk, Passage, Shelved
from ..domain.vectors import Vector


class Embedder(Protocol):
    """Places text in a space where distance stands in for likeness."""

    dimensions: int
    """How long the vectors are. A store filled by one embedder cannot be
    searched by another, and this is what lets that be caught rather than
    quietly returning nonsense."""

    min_score: float
    """Below this a match is noise rather than an answer. It belongs to the
    embedder because two models do not put related text at the same distance,
    so a threshold tuned for one is meaningless for the other."""

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch. One call for many texts, because providers charge per call."""
        ...


class VectorStore(Protocol):
    async def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    async def search(self, vector: Vector, limit: int) -> list[Passage]:
        """The closest passages, best first."""
        ...

    async def remove(self, document_id: str) -> int:
        """Forget a document. Returns how many passages went with it."""
        ...

    async def shelved(self) -> list[Shelved]:
        """Everything held, for an operator who wants to see it."""
        ...
