"""Which procedure this turn belongs to.

Ranking, not deciding. This module answers how well each procedure fits what
the customer just said; what to do about the answer is the nodes' business, and
keeping the two apart is what makes a threshold something you can argue about
without reading a graph.

`SopDefinition.examples` has carried the docstring "used for matching" since
the first phase without anything matching on it. This is that.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from itertools import islice

from ...domain.models import SopDefinition
from ...domain.vectors import Vector, cosine
from ...ports.knowledge import Embedder
from ...ports.stores import SopCatalog

MARGIN = 0.15
"""How far ahead a different procedure has to be before the conversation moves.

A conversation in progress is worth something. Without a margin, one sentence
that leaned the other way would bounce a customer between teammates mid-answer.
"""

AMBIGUITY = 0.08
"""Closer than this and the top two are a coin toss, so ask rather than guess.

"I need a refund for a damaged parcel" is genuinely both a return and a
delivery problem, and no threshold will make it one of them.
"""


@dataclass(frozen=True, slots=True)
class Match:
    sop: SopDefinition
    score: float


@dataclass(frozen=True, slots=True)
class Ranking:
    """The catalog ordered against one utterance, best first.

    The comparisons live here rather than in the nodes so that the arithmetic
    can be read and tested on its own, and so a node says what it decided
    instead of how it did the sums.
    """

    matches: tuple[Match, ...] = ()
    floor: float = 0.0
    """The embedder's own noise threshold, carried along so the comparisons below
    can be read without also being handed the embedder."""

    @property
    def best(self) -> Match | None:
        return self.matches[0] if self.matches else None

    @property
    def leaders(self) -> tuple[Match, ...]:
        """The procedures that matched at all, most likely first."""
        return tuple(match for match in self.matches if match.score >= self.floor)

    @property
    def ambiguous(self) -> bool:
        """Two procedures fit equally well, which is a question, not a decision."""
        front = self.leaders[:2]
        return len(front) == 2 and front[0].score - front[1].score < AMBIGUITY

    def score_of(self, sop_id: str) -> float:
        return next((m.score for m in self.matches if m.sop.sop_id == sop_id), 0.0)

    def beats(self, sop_id: str) -> Match | None:
        """A procedure that fits better than the one in progress, if there is one.

        Returns the winner rather than a flag, because every caller that wants
        to know also wants to know who.
        """
        best = self.best
        if best is None or best.sop.sop_id == sop_id or best.score < self.floor:
            return None
        return best if best.score - self.score_of(sop_id) >= MARGIN else None


class SopMatcher:
    """Ranks the catalog by how close it sits to what the customer said.

    The floor comes from the embedder rather than from here. "Is this
    similarity worth acting on" is a question about the model, not about what
    is being compared, and the answer is already calibrated once per embedder.
    """

    def __init__(self, catalog: SopCatalog, embedder: Embedder) -> None:
        self._catalog = catalog
        self._embedder = embedder
        self._known: tuple[tuple[SopDefinition, tuple[Vector, ...]], ...] | None = None
        self._lock = asyncio.Lock()

    async def rank(self, text: str) -> Ranking:
        known = await self._catalogue()
        if not text.strip() or not known:
            return Ranking(floor=self._embedder.min_score)

        query = (await self._embedder.embed([text]))[0]
        matches = [Match(sop, _closest(query, vectors)) for sop, vectors in known]
        matches.sort(key=lambda match: match.score, reverse=True)
        return Ranking(tuple(matches), floor=self._embedder.min_score)

    async def catch_all(self) -> SopDefinition | None:
        """The desk that takes a turn nothing else claimed."""
        return next((sop for sop in await self._catalog.list_all() if sop.catch_all), None)

    async def for_event(self, event: str) -> SopDefinition | None:
        """The procedure that answers this event, if one claims it.

        Looked up rather than ranked. An event arrives with a name, and
        guessing at a name is strictly worse than reading it.
        """
        return next((sop for sop in await self._catalog.list_all() if event in sop.events), None)

    async def _catalogue(self) -> tuple[tuple[SopDefinition, tuple[Vector, ...]], ...]:
        """Embed every procedure once.

        The catalog is fixed when the engine is built, so this is a warm-up
        rather than a cache with anything to invalidate. Doing it on first use
        instead of at startup keeps the ordering out of composition.
        """
        async with self._lock:
            if self._known is None:
                self._known = await self._embed_catalogue()
            return self._known

    async def _embed_catalogue(self) -> tuple[tuple[SopDefinition, tuple[Vector, ...]], ...]:
        sops = await self._catalog.list_all()
        spoken = [_phrases(sop) for sop in sops]
        # One call for the whole catalog, because a provider charges per call.
        vectors = iter(await self._embedder.embed([p for group in spoken for p in group]))
        return tuple(
            (sop, tuple(islice(vectors, len(group))))
            for sop, group in zip(sops, spoken, strict=True)
        )


def _phrases(sop: SopDefinition) -> list[str]:
    return [sop.description, *sop.examples]


def _closest(query: Vector, vectors: tuple[Vector, ...]) -> float:
    """The best single phrase, not the average of them all.

    A customer matches one example sharply. Averaging that against the four
    they said nothing like only buries the one signal there was.
    """
    return max((cosine(query, vector) for vector in vectors), default=0.0)
