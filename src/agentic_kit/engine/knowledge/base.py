"""Everything between a document arriving and a passage being quoted."""

from __future__ import annotations

import logging

from ...domain.knowledge import Chunk, Document, Passage, Shelved
from ...domain.text import readable
from ...errors import KnowledgeError
from ...ports.knowledge import Embedder, VectorStore
from ..guardrails.findings import Checkpoint
from ..guardrails.guard import Guardrails
from .chunker import chunk

TOP_K = 3
"""Passages per search. Three is what fits beside a procedure and a history
without the prompt becoming mostly quotation."""

logger = logging.getLogger("agentic_kit.knowledge")


class KnowledgeBase:
    """Ingests documents, finds passages, and distrusts the text both times.

    Screened twice on purpose. Ingestion is the cheap place to stop a poisoned
    document, since it happens once; retrieval is checked again because the
    store may hold passages put there before this check existed, or by a
    pipeline that is not this one.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        guardrails: Guardrails,
        *,
        top_k: int = TOP_K,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._guardrails = guardrails
        self._top_k = top_k

    async def ingest(self, document: Document) -> tuple[Chunk, ...]:
        """Store a document, or refuse the whole of it.

        Refused whole rather than paragraph by paragraph. A document that tells
        the agent what to do is not a good document with one bad passage, and
        keeping the parts that read innocently keeps the author's foothold.
        """
        text = readable(document.text)
        if not text.strip():
            raise KnowledgeError(f"{document.title!r} has no text to store")
        if not self._guardrails.check(Checkpoint.KNOWLEDGE, text).passed:
            raise KnowledgeError(
                f"{document.title!r} reads as an instruction to the agent rather than "
                "as reference material, so none of it was stored"
            )

        chunks = tuple(
            Chunk.of(document, position, piece) for position, piece in enumerate(chunk(text))
        )
        await self._store.add(chunks, await self._embedder.embed([c.text for c in chunks]))
        logger.info("ingested %s as %d passages", document.title, len(chunks))
        return chunks

    async def find(self, query: str) -> tuple[Passage, ...]:
        """The passages worth reading for this question, best first.

        An empty tuple is an ordinary answer: the library does not cover
        everything, and saying nothing beats quoting something unrelated.
        """
        if not query.strip():
            return ()

        vectors = await self._embedder.embed([query])
        found = await self._store.search(vectors[0], self._top_k)
        relevant = [p for p in found if p.score >= self._embedder.min_score]
        return tuple(p for p in relevant if self._trustworthy(p))

    async def forget(self, document_id: str) -> int:
        return await self._store.remove(document_id)

    async def shelf(self) -> list[Shelved]:
        return await self._store.shelved()

    def _trustworthy(self, passage: Passage) -> bool:
        """One passage dropped, the search kept.

        Unlike ingestion there is nobody to tell. The turn carries on with the
        passages that are left, and the log is how an operator finds out.
        """
        if self._guardrails.check(Checkpoint.KNOWLEDGE, passage.chunk.text).passed:
            return True
        logger.warning(
            "dropped passage %s from %s: reads as an instruction",
            passage.chunk.chunk_id,
            passage.chunk.source,
        )
        return False
